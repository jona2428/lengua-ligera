//! ¿Sabe cuándo no sabe fuera de su distribución? Confianza del motor entero (int8) sobre
//! Wikipedia vs texto barajado, invertido, ruido, código, y nombres propios reales vs inventados.
//! Uso: LLI=runs/tern32.lli EVAL_CHARS=60000 ood
use lengua_ligera::data::{encode, env, env_str, load_bin, load_vocab};
use lengua_ligera::intinfer::IntModel;
use lengua_ligera::rng::Rng;
use std::collections::HashMap;

/// (p_top Q16, acierto, etiqueta) por posición predicha.
fn score(im: &IntModel, ids: &[u8], tags: &[u8], threads: usize) -> Vec<(i32, bool, u8)> {
    let (t, v) = (im.ctx, im.vocab);
    let n_win = (ids.len() - 1) / t;
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|w| {
                s.spawn(move || {
                    let mut o = Vec::new();
                    for k in (w..n_win).step_by(threads) {
                        let off = k * t;
                        let lg = im.forward(&ids[off..off + t]);
                        for i in 0..t {
                            let p = im.probs_q16(&lg[i * v..(i + 1) * v]);
                            let am = (0..v).max_by_key(|&x| (p[x], std::cmp::Reverse(x))).unwrap();
                            o.push((p[am], am == ids[off + i + 1] as usize, tags[off + i + 1]));
                        }
                    }
                    o
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

fn line(name: &str, obs: &[(i32, bool, u8)]) {
    let n = obs.len().max(1) as f64;
    let acc = obs.iter().filter(|o| o.1).count() as f64 / n;
    let conf = obs.iter().map(|o| o.0 as f64).sum::<f64>() / n / 65536.0;
    let mut cols = String::new();
    for tau in [45875i32, 58982] {
        let a: Vec<_> = obs.iter().filter(|o| o.0 >= tau).collect();
        let ok = a.iter().filter(|o| o.1).count() as f64;
        cols += &format!(" | τ{:.0}%: responde {:5.1}%, acierta {:5.1}%", 100.0 * tau as f64 / 65536.0, 100.0 * a.len() as f64 / n, 100.0 * ok / a.len().max(1) as f64);
    }
    println!("{name:<30} n {:>6} | acierto {:5.1}% | confianza media {:5.1}%{cols}", obs.len(), 100.0 * acc, 100.0 * conf);
}

fn pseudo_word(rng: &mut Rng) -> String {
    let (c, v) = ("bcdfglmnprstvz".as_bytes(), "aeiou".as_bytes());
    let mut s = String::new();
    for _ in 0..2 + rng.below(3) {
        s.push(c[rng.below(c.len())] as char);
        s.push(v[rng.below(v.len())] as char);
        if rng.below(4) == 0 {
            s.push(c[rng.below(c.len())] as char);
        }
    }
    let mut ch = s.chars();
    let f = ch.next().unwrap().to_ascii_uppercase();
    std::iter::once(f).chain(ch).collect()
}

/// Nombres propios: palabras con mayúscula inicial, ≥3 letras, que no empiezan oración.
/// Devuelve el texto (con nombres reemplazados si `invent`) y, por carácter, el id del nombre (0 = ninguno).
fn names(text: &[char], invent: bool, rng: &mut Rng) -> (Vec<char>, Vec<u32>) {
    let mut out = Vec::new();
    let mut nid = Vec::new();
    let mut map: HashMap<String, (String, u32)> = HashMap::new();
    let mut i = 0;
    while i < text.len() {
        let start_word = i == 0 || !text[i - 1].is_alphanumeric();
        if start_word && text[i].is_uppercase() {
            let mut j = i;
            while j < text.len() && text[j].is_alphabetic() {
                j += 1;
            }
            let prev = text[..i].iter().rev().find(|c| !c.is_whitespace());
            let sentence_start = matches!(prev, None | Some('.') | Some('\n') | Some(':') | Some('?') | Some('!'));
            if j - i >= 3 && !sentence_start {
                let w: String = text[i..j].iter().collect();
                let k = map.len() as u32 + 1;
                let (rep, id) = map.entry(w.clone()).or_insert_with(|| (if invent { pseudo_word(rng) } else { w.clone() }, k)).clone();
                for ch in rep.chars() {
                    out.push(ch);
                    nid.push(id);
                }
                i = j;
                continue;
            }
        }
        out.push(text[i]);
        nid.push(0);
        i += 1;
    }
    (out, nid)
}

/// Etiqueta por carácter de nombre (desde su 2º carácter): 1 = primera mención en la ventana, 2 = repetida.
fn name_tags(nid: &[u32], ctx: usize) -> Vec<u8> {
    let mut tags = vec![0u8; nid.len()];
    for w0 in (0..nid.len()).step_by(ctx) {
        let mut seen = std::collections::HashSet::new();
        let mut i = w0;
        let end = (w0 + ctx + 1).min(nid.len());
        while i < end {
            if nid[i] == 0 {
                i += 1;
                continue;
            }
            let id = nid[i];
            let first = !seen.contains(&id);
            let mut j = i + 1;
            while j < end && nid[j] == id {
                tags[j] = if first { 1 } else { 2 };
                j += 1;
            }
            seen.insert(id);
            i = j;
        }
    }
    tags
}

fn main() {
    let mut im = IntModel::load(&env_str("LLI", "runs/tern32.lli"));
    im.act_bits = env("ACT_BITS", 8);
    let threads: usize = env("THREADS", 2);
    let n: usize = env("EVAL_CHARS", 60_000);
    let vocab = load_vocab("data/vocab.txt");
    let valid = load_bin("data/valid.bin");
    let wiki: Vec<char> = valid[..n + 1].iter().map(|&i| vocab[i as usize]).collect();
    let mut rng = Rng::new(7);
    let enc = |cs: &[char]| encode(&vocab, &cs.iter().collect::<String>());
    let none = |len: usize| vec![0u8; len];

    println!("motor entero act_bits {} — confianza = p del carácter más probable (Q16, sólo enteros)\n", im.act_bits);
    let ids = valid[..n + 1].to_vec();
    line("Wikipedia (misma distribución)", &score(&im, &ids, &none(ids.len()), threads));

    let words: Vec<String> = wiki.iter().collect::<String>().split(' ').map(String::from).collect();
    let mut sh = words.clone();
    for i in (1..sh.len()).rev() {
        sh.swap(i, rng.below(i + 1));
    }
    let ids = encode(&vocab, &sh.join(" "));
    line("palabras barajadas", &score(&im, &ids, &none(ids.len()), threads));

    let rev: Vec<char> = wiki.iter().rev().cloned().collect();
    let ids = enc(&rev);
    line("texto invertido", &score(&im, &ids, &none(ids.len()), threads));

    let ids: Vec<u8> = (0..=n).map(|_| valid[rng.below(valid.len())]).collect();
    line("ruido (unigrama del español)", &score(&im, &ids, &none(ids.len()), threads));

    let mut code = String::new();
    for f in ["src/intinfer.rs", "src/model.rs", "src/quant.rs", "src/ops.rs", "src/bin/train.rs", "src/export.rs"] {
        code += &std::fs::read_to_string(f).unwrap();
    }
    let ids = encode(&vocab, &code);
    line("código Rust (otro dominio)", &score(&im, &ids, &none(ids.len()), threads));

    println!("\nnombres propios (desde su 2º carácter):");
    for (invent, label) in [(false, "reales"), (true, "inventados")] {
        let (txt, nid) = names(&wiki, invent, &mut Rng::new(11));
        let ids = enc(&txt);
        let tags = name_tags(&nid, im.ctx);
        let obs = score(&im, &ids, &tags, threads);
        line(&format!("{label}, 1ª mención en ventana"), &obs.iter().filter(|o| o.2 == 1).cloned().collect::<Vec<_>>());
        line(&format!("{label}, mención repetida"), &obs.iter().filter(|o| o.2 == 2).cloned().collect::<Vec<_>>());
    }
}
