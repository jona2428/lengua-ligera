//! Curva calidad vs bits por peso: cuantiza todas las matrices del modelo con cada
//! formato y mide bits por carácter en validación.
//! Uso: MODEL=runs/base.llg EVAL_CHARS=500000 quant_curve [filtro]
use lengua_ligera::data::{env, env_str, load_bin, load_model};
use lengua_ligera::eval::{bpc, evaluate};
use lengua_ligera::quant::catalog;


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
    for f in catalog() {
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
