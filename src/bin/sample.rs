//! Genera texto. Uso: MODEL=runs/base.llg N=400 TEMP=0.8 FMT=... sample "prompt"
use lengua_ligera::data::{decode, encode, env, env_str, load_model, load_vocab};
use lengua_ligera::model::Acts;
use lengua_ligera::rng::Rng;

fn main() {
    let m = load_model(&env_str("MODEL", "runs/base.llg"));
    let vocab = load_vocab("data/vocab.txt");
    let n: usize = env("N", 400);
    let temp: f32 = env("TEMP", 0.8);
    let mut rng = Rng::new(env("SEED", 42));
    let prompt = std::env::args().nth(1).unwrap_or_else(|| "La ciudad de".to_string());
    let mut ids = encode(&vocab, &prompt);
    let wt = m.transposed();
    let mut a = Acts::new(&m.cfg);
    let v = m.cfg.vocab;
    for _ in 0..n {
        let start = ids.len().saturating_sub(m.cfg.ctx);
        let ctx = &ids[start..];
        m.forward(&wt, &mut a, ctx, None);
        let row = &a.probs[(ctx.len() - 1) * v..ctx.len() * v];
        let w: Vec<f64> = row.iter().map(|p| (*p as f64).powf(1.0 / temp as f64)).collect();
        let z: f64 = w.iter().sum();
        let mut r = rng.f32() as f64 * z;
        let mut next = v - 1;
        for (i, wi) in w.iter().enumerate() {
            r -= wi;
            if r <= 0.0 {
                next = i;
                break;
            }
        }
        ids.push(next as u8);
    }
    println!("{}", decode(&vocab, &ids));
}
