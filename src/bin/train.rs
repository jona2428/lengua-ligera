//! Entrena el modelo FP32. Configuración por variables de entorno:
//! D L H CTX BATCH STEPS LR WARMUP WD SEED EVAL_EVERY EVAL_CHARS OUT THREADS
use lengua_ligera::data::{env, env_str, load_bin, save_model};
use lengua_ligera::model::{Acts, Config, Model};
use lengua_ligera::rng::Rng;
use std::time::Instant;

struct Worker {
    acts: Acts,
    grad: Vec<f32>,
    loss: f64,
}

/// Pérdida media (nats/carácter) sobre ventanas disjuntas del set de validación.
pub fn evaluate(m: &Model, valid: &[u8], chars: usize, threads: usize) -> f64 {
    let t = m.cfg.ctx;
    let n_win = (chars.min(valid.len() - 1)) / t;
    let wt = m.transposed();
    let sums: Vec<f64> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|w| {
                let wt = &wt;
                s.spawn(move || {
                    let mut a = Acts::new(&m.cfg);
                    let mut sum = 0f64;
                    for k in (w..n_win).step_by(threads) {
                        let o = k * t;
                        sum += m.forward(wt, &mut a, &valid[o..o + t], Some(&valid[o + 1..o + t + 1])) as f64;
                    }
                    sum
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    sums.iter().sum::<f64>() / n_win as f64
}

fn main() {
    let cfg = Config {
        vocab: 128,
        ctx: env("CTX", 128),
        d: env("D", 256),
        layers: env("L", 4),
        heads: env("H", 4),
    };
    let batch: usize = env("BATCH", 48);
    let steps: usize = env("STEPS", 10000);
    let lr_max: f32 = env("LR", 1e-3);
    let warmup: usize = env("WARMUP", 300);
    let wd: f32 = env("WD", 0.1);
    let seed: u64 = env("SEED", 1);
    let eval_every: usize = env("EVAL_EVERY", 500);
    let eval_chars: usize = env("EVAL_CHARS", 500_000);
    let out = env_str("OUT", "runs/base.llg");
    let threads: usize = env("THREADS", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));

    let train = load_bin("data/train.bin");
    let valid = load_bin("data/valid.bin");
    let mut m = Model::init(cfg, seed);
    let np = m.n_params();
    println!("config {cfg:?} params {np} batch {batch} steps {steps} lr {lr_max} threads {threads}");
    if let Some(dir) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(dir).ok();
    }

    // decaimiento sólo en matrices
    let mut decay = vec![0f32; np];
    for mat in m.lay.matrices(&cfg) {
        decay[mat.off..mat.off + mat.rows * mat.cols].fill(1.0);
    }
    let mut mom = vec![0f32; np];
    let mut vel = vec![0f32; np];
    let mut grad = vec![0f32; np];
    let mut workers: Vec<Worker> = (0..threads)
        .map(|_| Worker { acts: Acts::new(&cfg), grad: vec![0f32; np], loss: 0.0 })
        .collect();
    let mut rng = Rng::new(seed ^ 0xABCDEF);
    let (b1, b2, eps) = (0.9f32, 0.95f32, 1e-8f32);
    let t = cfg.ctx;
    let gscale = 1.0 / (batch * t) as f32;
    let mut best = f64::INFINITY;
    let t0 = Instant::now();
    let mut ema_loss = 0f64;
    let mut tm = [0f64; 4];
    let print_every: usize = env("PRINT_EVERY", 50);

    for step in 1..=steps {
        let offs: Vec<usize> = (0..batch).map(|_| rng.below(train.len() - t - 1)).collect();
        let ta = Instant::now();
        let wt = m.transposed();
        let tb = Instant::now();
        std::thread::scope(|s| {
            for (wi, w) in workers.iter_mut().enumerate() {
                let (m, wt, offs, train) = (&m, &wt, &offs, &train);
                s.spawn(move || {
                    w.grad.fill(0.0);
                    w.loss = 0.0;
                    for bi in (wi..offs.len()).step_by(threads) {
                        let seq = &train[offs[bi]..offs[bi] + t + 1];
                        w.loss += m.forward(wt, &mut w.acts, &seq[..t], Some(&seq[1..])) as f64;
                        m.backward(wt, &mut w.acts, &seq[..t], &seq[1..], &mut w.grad, gscale);
                    }
                });
            }
        });
        let tc = Instant::now();
        // reducción paralela de gradientes
        let chunk = np.div_ceil(threads);
        std::thread::scope(|s| {
            for (ci, gc) in grad.chunks_mut(chunk).enumerate() {
                let workers = &workers;
                s.spawn(move || {
                    let o = ci * chunk;
                    let n = gc.len();
                    gc.copy_from_slice(&workers[0].grad[o..o + n]);
                    for w in &workers[1..] {
                        for (a, b) in gc.iter_mut().zip(&w.grad[o..o + n]) {
                            *a += b;
                        }
                    }
                });
            }
        });
        let td = Instant::now();
        let loss = workers.iter().map(|w| w.loss).sum::<f64>() / batch as f64;
        ema_loss = if step == 1 { loss } else { 0.98 * ema_loss + 0.02 * loss };

        // recorte de norma + AdamW
        let gnorm = grad.iter().map(|g| (*g as f64) * (*g as f64)).sum::<f64>().sqrt() as f32;
        let clip = if gnorm > 1.0 { 1.0 / gnorm } else { 1.0 };
        let lr = if step <= warmup {
            lr_max * step as f32 / warmup as f32
        } else {
            let pr = (step - warmup) as f32 / (steps - warmup).max(1) as f32;
            lr_max * (0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * pr).cos()))
        };
        let (bc1, bc2) = (1.0 - b1.powi(step as i32), 1.0 - b2.powi(step as i32));
        for i in 0..np {
            let gi = grad[i] * clip;
            mom[i] = b1 * mom[i] + (1.0 - b1) * gi;
            vel[i] = b2 * vel[i] + (1.0 - b2) * gi * gi;
            let upd = (mom[i] / bc1) / ((vel[i] / bc2).sqrt() + eps);
            m.p[i] -= lr * (upd + wd * decay[i] * m.p[i]);
        }

        let te = Instant::now();
        tm[0] += (tb - ta).as_secs_f64(); tm[1] += (tc - tb).as_secs_f64(); tm[2] += (td - tc).as_secs_f64(); tm[3] += (te - td).as_secs_f64();
        if step % print_every == 0 || step == 1 {
            let el = t0.elapsed().as_secs_f64();
            let tps = (step * batch * t) as f64 / el;
            println!(
                "step {step:6} loss {loss:.4} ema {ema_loss:.4} bpc~{:.3} gnorm {gnorm:.2} lr {lr:.2e} tok/s {tps:.0} t {el:.0}s [tr {:.2} fb {:.2} red {:.2} opt {:.2}]",
                ema_loss / std::f64::consts::LN_2, tm[0], tm[1], tm[2], tm[3]
            );
        }
        if step % eval_every == 0 || step == steps {
            let vl = evaluate(&m, &valid, eval_chars, threads);
            let bpc = vl / std::f64::consts::LN_2;
            let mark = if vl < best {
                best = vl;
                save_model(&out, &m);
                " *guardado"
            } else {
                ""
            };
            println!("EVAL step {step} valid_loss {vl:.4} valid_bpc {bpc:.4} ppl_char {:.3}{mark}", vl.exp());
        }
    }
    println!("fin: mejor valid_bpc {:.4} en {:.0}s", best / std::f64::consts::LN_2, t0.elapsed().as_secs_f64());
}
