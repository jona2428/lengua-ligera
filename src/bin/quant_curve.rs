//! Curva calidad vs bits por peso: cuantiza todas las matrices del modelo con cada
//! formato y mide bits por carácter en validación.
//! Uso: MODEL=runs/base.llg EVAL_CHARS=500000 quant_curve [filtro]
use lengua_ligera::data::{env, env_str, load_bin, load_model};
use lengua_ligera::eval::{bpc, evaluate};
use lengua_ligera::quant::Fmt;

fn formats() -> Vec<Fmt> {
    use Fmt::*;
    let mut v = vec![F32, Float { e: 5, m: 10 }, Float { e: 8, m: 7 }];
    for g in [32usize, 128] {
        v.push(Int { bits: 8, group: g });
        v.push(MiniFloat { e: 4, m: 3, group: g });
        v.push(MiniFloat { e: 5, m: 2, group: g });
        v.push(MiniFloat { e: 3, m: 2, group: g });
        v.push(MiniFloat { e: 2, m: 3, group: g });
        v.push(Int { bits: 6, group: g });
        v.push(Int { bits: 5, group: g });
        v.push(Int { bits: 4, group: g });
        v.push(MiniFloat { e: 2, m: 1, group: g });
        v.push(MiniFloat { e: 3, m: 0, group: g });
        v.push(MiniFloat { e: 1, m: 2, group: g });
        v.push(Nf4 { group: g });
        v.push(KMeans { bits: 4, group: g });
        v.push(Int { bits: 3, group: g });
        v.push(MiniFloat { e: 1, m: 1, group: g });
        v.push(MiniFloat { e: 2, m: 0, group: g });
        v.push(KMeans { bits: 3, group: g });
        v.push(Int { bits: 2, group: g });
        v.push(KMeans { bits: 2, group: g });
        v.push(Ternary { group: g });
        v.push(Binary { group: g });
    }
    v.push(Int { bits: 8, group: 0 });
    v.push(Int { bits: 4, group: 0 });
    v
}

fn main() {
    let path = env_str("MODEL", "runs/base.llg");
    let chars: usize = env("EVAL_CHARS", 500_000);
    let threads: usize = env("THREADS", 6);
    let filter = std::env::args().nth(1).unwrap_or_default();
    let base = load_model(&path);
    let valid = load_bin("data/valid.bin");
    let mats = base.lay.matrices(&base.cfg);
    let nq: usize = mats.iter().map(|m| m.rows * m.cols).sum();
    let rest = base.p.len() - nq;
    println!("modelo {path} params {} (matrices {nq}, resto {rest} en fp16)", base.p.len());
    println!("{:<22} {:>7} {:>9} {:>9} {:>10}", "formato", "bits", "bpc", "Δbpc", "KB");
    let mut ref_bpc = None;
    for f in formats() {
        let name = f.name();
        if !filter.is_empty() && !name.contains(&filter) && name != "fp32" {
            continue;
        }
        let mut m = load_model(&path);
        for mat in &mats {
            f.apply(&mut m.p[mat.off..mat.off + mat.rows * mat.cols]);
        }
        let b = bpc(evaluate(&m, &valid, chars, threads));
        let r = *ref_bpc.get_or_insert(b);
        let kb = (nq as f64 * f.bits() + rest as f64 * 16.0) / 8.0 / 1024.0;
        println!("{name:<22} {:>7.3} {b:>9.4} {:>+9.4} {kb:>10.1}", f.bits(), b - r);
    }
}
