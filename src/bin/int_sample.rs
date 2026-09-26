//! Genera texto sólo con el motor entero, cargando únicamente el archivo .lli.
//! Uso: LLI=runs/tern32.lli N=400 TEMP_NUM=7 TEMP_DEN=10 int_sample "prompt"
use lengua_ligera::data::{decode, encode, env, env_str, load_vocab};
use lengua_ligera::intinfer::IntModel;

fn main() {
    let mut im = IntModel::load(&env_str("LLI", "runs/tern32.lli"));
    im.act_bits = env("ACT_BITS", 0);
    let vocab = load_vocab("data/vocab.txt");
    let n: usize = env("N", 400);
    let (tn, td): (i64, i64) = (env("TEMP_NUM", 7), env("TEMP_DEN", 10));
    let mut rng: u64 = env("SEED", 42u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let prompt = std::env::args().nth(1).unwrap_or_else(|| "La ciudad de".to_string());
    let mut ids = encode(&vocab, &prompt);
    for _ in 0..n {
        let start = ids.len().saturating_sub(im.ctx);
        let ctx = &ids[start..];
        let logits = im.forward(ctx);
        let last = &logits[(ctx.len() - 1) * im.vocab..ctx.len() * im.vocab];
        ids.push(im.sample(last, tn, td, &mut rng));
    }
    println!("{}", decode(&vocab, &ids));
}
