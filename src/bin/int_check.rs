//! Exporta un checkpoint QAT al motor entero (.lli), lo recarga desde disco y compara
//! contra el motor float con los mismos pesos cuantizados.
//! Uso: MODEL=runs/qat_tern32.llg FMT=ternario-g32 OUT=runs/tern32.lli EVAL_CHARS=500000 int_check
use lengua_ligera::data::{env, env_str, load_bin, load_model};
use lengua_ligera::export::export;
use lengua_ligera::intinfer::IntModel;
use lengua_ligera::model::Acts;
use lengua_ligera::quant::Fmt;
use std::time::Instant;

fn main() {
    let path = env_str("MODEL", "runs/qat_tern32.llg");
    let fname = env_str("FMT", "ternario-g32");
    let out = env_str("OUT", "runs/tern32.lli");
    let chars: usize = env("EVAL_CHARS", 500_000);
    let threads: usize = env("THREADS", 6);
    let f = Fmt::parse(&fname).expect("formato desconocido");

    let mut fm = load_model(&path);
    let im0 = export(&fm, &f);
    im0.save(&out);
    let bytes = std::fs::metadata(&out).unwrap().len();
    let mut im = IntModel::load(&out);
    im.act_bits = env("ACT_BITS", 0);
    println!("activaciones: {}", if im.act_bits == 0 { "Q16 (i32)".to_string() } else { format!("int{}", im.act_bits) });
    println!("exportado {out}: {:.1} KB (sólo enteros)", bytes as f64 / 1024.0);
    for mat in fm.lay.matrices(&fm.cfg) {
        f.apply(&mut fm.p[mat.off..mat.off + mat.rows * mat.cols]);
    }
    let valid = load_bin("data/valid.bin");
    let t = fm.cfg.ctx;
    let v = fm.cfg.vocab;
    let n_win = chars.min(valid.len() - 1) / t;
    let wt = fm.transposed();
    let t0 = Instant::now();
    // (nats float, nats entero, coincidencias argmax, posiciones, max |Δlogit|)
    let res: Vec<(f64, f64, usize, usize, f64)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|w| {
                let (fm, im, wt, valid) = (&fm, &im, &wt, &valid);
                s.spawn(move || {
                    let mut a = Acts::new(&fm.cfg);
                    let (mut lf, mut li, mut agree, mut n, mut maxd) = (0f64, 0f64, 0usize, 0usize, 0f64);
                    for k in (w..n_win).step_by(threads) {
                        let o = k * t;
                        let inp = &valid[o..o + t];
                        let tgt = &valid[o + 1..o + t + 1];
                        lf += fm.forward(wt, &mut a, inp, Some(tgt)) as f64 * t as f64;
                        let il = im.forward(inp);
                        for i in 0..t {
                            let row: Vec<f64> = il[i * v..(i + 1) * v].iter().map(|x| *x as f64 / 65536.0).collect();
                            let mx = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            let lse = mx + row.iter().map(|x| (x - mx).exp()).sum::<f64>().ln();
                            li -= row[tgt[i] as usize] - lse;
                            let am_i = (0..v).max_by(|&x, &y| row[x].partial_cmp(&row[y]).unwrap()).unwrap();
                            let pf = &a.probs[i * v..(i + 1) * v];
                            let am_f = (0..v).max_by(|&x, &y| pf[x].partial_cmp(&pf[y]).unwrap()).unwrap();
                            agree += (am_i == am_f) as usize;
                            n += 1;
                            // compara log-probabilidades
                            for j in 0..v {
                                let d = ((row[j] - lse) - (pf[j].max(1e-30) as f64).ln()).abs();
                                if pf[j] > 1e-4 && d > maxd {
                                    maxd = d;
                                }
                            }
                        }
                    }
                    (lf, li, agree, n, maxd)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (lf, li, ag, n) = res.iter().fold((0f64, 0f64, 0usize, 0usize), |a, r| (a.0 + r.0, a.1 + r.1, a.2 + r.2, a.3 + r.3));
    let maxd = res.iter().map(|r| r.4).fold(0f64, f64::max);
    let ln2 = std::f64::consts::LN_2;
    println!("posiciones {n} en {:.0}s", t0.elapsed().as_secs_f64());
    println!("motor float  (pesos {fname}): {:.4} bpc", lf / n as f64 / ln2);
    println!("motor entero (archivo .lli):  {:.4} bpc", li / n as f64 / ln2);
    println!("mismo carácter más probable: {:.3}%  | max |Δ log p| (p>1e-4): {maxd:.4}", 100.0 * ag as f64 / n as f64);
}
