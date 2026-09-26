//! ¿Sabe cuándo no sabe? Calibración y abstención de la predicción del siguiente carácter.
//! La confianza del motor entero (p_top en Q16) se calcula sólo con enteros; las métricas
//! (ECE, AUROC, curva cobertura/precisión) se reportan en f64 como medición, fuera del motor.
//! Uso: LLI=runs/tern32.lli MODEL=runs/qat_tern32.llg FMT=ternario-g32 BASE=runs/ctrl_fp32.llg EVAL_CHARS=200000 honesty
use lengua_ligera::data::{env, env_str, load_bin, load_model};
use lengua_ligera::intinfer::IntModel;
use lengua_ligera::model::{Acts, Model};
use lengua_ligera::quant::Fmt;

/// (confianza top-1, acierto) por posición.
type Obs = Vec<(f64, bool)>;

fn run_float(m: &Model, valid: &[u8], n_win: usize, threads: usize) -> Obs {
    let (t, v) = (m.cfg.ctx, m.cfg.vocab);
    let wt = m.transposed();
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|w| {
                let wt = &wt;
                s.spawn(move || {
                    let mut a = Acts::new(&m.cfg);
                    let mut o = Vec::new();
                    for k in (w..n_win).step_by(threads) {
                        let off = k * t;
                        m.forward(wt, &mut a, &valid[off..off + t], None);
                        for i in 0..t {
                            let r = &a.probs[i * v..(i + 1) * v];
                            let am = (0..v).max_by(|&x, &y| r[x].partial_cmp(&r[y]).unwrap()).unwrap();
                            o.push((r[am] as f64, am == valid[off + i + 1] as usize));
                        }
                    }
                    o
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

fn run_int(im: &IntModel, valid: &[u8], n_win: usize, threads: usize) -> (Obs, Vec<(i32, bool)>) {
    let (t, v) = (im.ctx, im.vocab);
    let raw: Vec<(i32, bool)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|w| {
                s.spawn(move || {
                    let mut o = Vec::new();
                    for k in (w..n_win).step_by(threads) {
                        let off = k * t;
                        let lg = im.forward(&valid[off..off + t]);
                        for i in 0..t {
                            let p = im.probs_q16(&lg[i * v..(i + 1) * v]);
                            let am = (0..v).max_by_key(|&x| (p[x], std::cmp::Reverse(x))).unwrap();
                            o.push((p[am], am == valid[off + i + 1] as usize));
                        }
                    }
                    o
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    (raw.iter().map(|&(p, c)| (p as f64 / 65536.0, c)).collect(), raw)
}

fn report(name: &str, obs: &Obs) {
    let n = obs.len() as f64;
    let acc = obs.iter().filter(|o| o.1).count() as f64 / n;
    // ECE con 15 intervalos
    let mut bins = [(0f64, 0f64, 0f64); 15];
    for &(p, c) in obs {
        let b = ((p * 15.0) as usize).min(14);
        bins[b].0 += 1.0;
        bins[b].1 += p;
        bins[b].2 += c as u8 as f64;
    }
    let ece: f64 = bins.iter().filter(|b| b.0 > 0.0).map(|b| (b.1 - b.2).abs() / n).sum();
    // orden por confianza descendente → curva cobertura/precisión y AUROC
    let mut s = obs.clone();
    s.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let (npos, nneg) = (s.iter().filter(|o| o.1).count() as f64, s.iter().filter(|o| !o.1).count() as f64);
    let (mut seen_neg, mut auc) = (0f64, 0f64);
    for o in s.iter().rev() {
        if o.1 {
            auc += seen_neg;
        } else {
            seen_neg += 1.0;
        }
    }
    auc /= npos * nneg;
    let mut cov_at = [0f64; 3];
    let targets = [0.90, 0.95, 0.99];
    let mut good = 0f64;
    let mut acc_half = 0f64;
    for (i, o) in s.iter().enumerate() {
        good += o.1 as u8 as f64;
        let k = (i + 1) as f64;
        for (j, &tg) in targets.iter().enumerate() {
            if good / k >= tg && k >= 100.0 {
                cov_at[j] = k / n;
            }
        }
        if i + 1 == s.len() / 2 {
            acc_half = good / k;
        }
    }
    println!(
        "{name:<26} acierto {:.2}% | ECE {:.4} | AUROC {:.4} | acierto al 50% cobertura {:.2}% | cobertura con ≥90% {:.1}%, ≥95% {:.1}%, ≥99% {:.1}%",
        100.0 * acc, ece, auc, 100.0 * acc_half, 100.0 * cov_at[0], 100.0 * cov_at[1], 100.0 * cov_at[2]
    );
}

fn main() {
    let valid = load_bin("data/valid.bin");
    let chars: usize = env("EVAL_CHARS", 200_000);
    let threads: usize = env("THREADS", 2);
    let base = load_model(&env_str("BASE", "runs/ctrl_fp32.llg"));
    let n_win = chars.min(valid.len() - 1) / base.cfg.ctx;
    report("float FP32 (control)", &run_float(&base, &valid, n_win, threads));
    let mut q = load_model(&env_str("MODEL", "runs/qat_tern32.llg"));
    let f = Fmt::parse(&env_str("FMT", "ternario-g32")).unwrap();
    for mat in q.lay.matrices(&q.cfg) {
        f.apply(&mut q.p[mat.off..mat.off + mat.rows * mat.cols]);
    }
    report("float, pesos ternarios", &run_float(&q, &valid, n_win, threads));
    let mut im = IntModel::load(&env_str("LLI", "runs/tern32.lli"));
    for ab in [0u32, 8] {
        im.act_bits = ab;
        let (obs, raw) = run_int(&im, &valid, n_win, threads);
        report(if ab == 0 { "entero Q16" } else { "entero int8" }, &obs);
        if ab == 8 {
            // abstención con umbral entero: responde sólo si p_top ≥ τ (Q16)
            for tau in [32768i32, 45875, 58982] {
                let ans: Vec<_> = raw.iter().filter(|r| r.0 >= tau).collect();
                let ok = ans.iter().filter(|r| r.1).count();
                println!(
                    "  τ = {tau}/65536 ({:.0}%): responde {:.1}% de las veces, acierta {:.2}% de lo que responde",
                    100.0 * tau as f64 / 65536.0, 100.0 * ans.len() as f64 / raw.len() as f64, 100.0 * ok as f64 / ans.len().max(1) as f64
                );
            }
        }
    }
}
