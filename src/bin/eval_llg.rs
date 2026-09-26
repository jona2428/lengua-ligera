//! Pérdida en valid de un .llg (opcionalmente con pesos cuantizados). Referencia de paridad
//! para kaggle/molde.py. Uso: MODEL=runs/base.llg FMT=ternario-g32 EVAL_CHARS=100000 eval_llg
use lengua_ligera::data::{env, env_str, load_bin, load_model};
use lengua_ligera::eval::evaluate;
use lengua_ligera::quant::Fmt;

fn main() {
    let mut m = load_model(&env_str("MODEL", "runs/base.llg"));
    let fmt = env_str("FMT", "");
    if !fmt.is_empty() {
        let f = Fmt::parse(&fmt).expect("formato desconocido");
        for mat in m.lay.matrices(&m.cfg) {
            f.apply(&mut m.p[mat.off..mat.off + mat.rows * mat.cols]);
        }
    }
    let valid = load_bin("data/valid.bin");
    let vl = evaluate(&m, &valid, env("EVAL_CHARS", 100_000), env("THREADS", 4));
    println!("Rust: valid_loss {vl:.6} valid_bpc {:.6}", vl / std::f64::consts::LN_2);
}
