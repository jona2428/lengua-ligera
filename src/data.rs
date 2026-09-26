//! Corpus, vocabulario y checkpoints.
use crate::model::{Config, Model};
use std::io::{Read, Write};

pub fn load_bin(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("no pude leer {path}: {e}"))
}

/// vocab.txt: un string JSON por línea (formato de json.dumps de Python).
pub fn load_vocab(path: &str) -> Vec<char> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("no pude leer {path}: {e}"));
    text.lines().filter(|l| !l.is_empty()).map(parse_json_char).collect()
}

fn parse_json_char(s: &str) -> char {
    let inner = &s[1..s.len() - 1];
    let b = inner.as_bytes();
    if b[0] != b'\\' {
        return inner.chars().next().unwrap();
    }
    match b[1] {
        b'n' => '\n',
        b't' => '\t',
        b'r' => '\r',
        b'"' => '"',
        b'\\' => '\\',
        b'/' => '/',
        b'u' => {
            let hi = u32::from_str_radix(&inner[2..6], 16).unwrap();
            if (0xD800..0xDC00).contains(&hi) {
                let lo = u32::from_str_radix(&inner[8..12], 16).unwrap();
                char::from_u32(0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)).unwrap()
            } else {
                char::from_u32(hi).unwrap_or('\u{FFFD}')
            }
        }
        _ => '\u{FFFD}',
    }
}

pub fn encode(vocab: &[char], s: &str) -> Vec<u8> {
    let unk = (vocab.len() - 1) as u8;
    s.chars()
        .map(|c| vocab.iter().position(|&v| v == c).map(|i| i as u8).unwrap_or(unk))
        .collect()
}

pub fn decode(vocab: &[char], ids: &[u8]) -> String {
    ids.iter().map(|&i| vocab[i as usize]).collect()
}

const MAGIC: &[u8; 4] = b"LLG1";

pub fn save_model(path: &str, m: &Model) {
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(MAGIC).unwrap();
    let c = m.cfg;
    for v in [c.vocab, c.ctx, c.d, c.layers, c.heads] {
        f.write_all(&(v as u32).to_le_bytes()).unwrap();
    }
    let mut buf = Vec::with_capacity(m.p.len() * 4);
    for x in &m.p {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    f.write_all(&buf).unwrap();
}

pub fn load_model(path: &str) -> Model {
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("no pude abrir {path}: {e}"));
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes).unwrap();
    assert_eq!(&bytes[..4], MAGIC, "checkpoint inválido");
    let u = |i: usize| u32::from_le_bytes(bytes[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
    let cfg = Config { vocab: u(0), ctx: u(1), d: u(2), layers: u(3), heads: u(4) };
    let mut m = Model::zeros(cfg);
    let body = &bytes[24..];
    assert_eq!(body.len(), m.p.len() * 4, "tamaño de checkpoint no calza con config");
    for (x, ch) in m.p.iter_mut().zip(body.chunks_exact(4)) {
        *x = f32::from_le_bytes(ch.try_into().unwrap());
    }
    m
}

pub fn env<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

pub fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}
