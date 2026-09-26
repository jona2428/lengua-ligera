//! Motor de inferencia 100% entero. Este módulo no usa f32 ni f64.
//!
//! - Pesos de matrices: trits i8 ∈ {-1, 0, +1} con una escala entera i16 por bloque
//!   (exponente `sh` por matriz: valor real = t · s / 2^sh).
//! - Activaciones: punto fijo Q16 en i32; acumuladores en i64.
//! - LayerNorm: media/varianza enteras + raíz cuadrada entera.
//! - Softmax: exp2 por tabla entera. GELU: tabla entera con interpolación lineal.
use std::io::{Read, Write};

pub const Q: u32 = 16;
/// 1e-5 en Q32 (épsilon de LayerNorm).
const LN_EPS_Q32: i64 = 42_950;
/// log2(e) en Q16.
const LOG2E_Q16: i64 = 94_548;
pub const EXP2_BITS: u32 = 10;
pub const GELU_RANGE: i64 = 8; // tabla cubre [-8, 8)
pub const GELU_STEP_BITS: u32 = 8; // paso 1/256

/// Matriz ternaria/binaria con escala por bloque a lo largo de las columnas.
pub struct TMat {
    pub rows: usize,
    pub cols: usize,
    pub group: usize,
    pub sh: i32,
    pub t: Vec<i8>,
    pub s: Vec<i16>,
}

#[inline]
fn to_q32(v: i64, sh: i32) -> i64 {
    // v está en Q(16+sh)
    if sh >= 16 {
        v >> (sh - 16)
    } else {
        v << (16 - sh)
    }
}

impl TMat {
    /// y[t×cols] (Q32) += x[t×rows] (Q16) · W
    fn mul_acc(&self, y: &mut [i64], x: &[i32], t: usize) {
        let (k, n, g) = (self.rows, self.cols, self.group);
        let nb = n / g;
        for i in 0..t {
            let yr = &mut y[i * n..(i + 1) * n];
            for kk in 0..k {
                let xv = x[i * k + kk] as i64;
                if xv == 0 {
                    continue;
                }
                let tr = &self.t[kk * n..(kk + 1) * n];
                let sr = &self.s[kk * nb..(kk + 1) * nb];
                for b in 0..nb {
                    let xs = to_q32(xv * sr[b] as i64, self.sh);
                    let yb = &mut yr[b * g..(b + 1) * g];
                    let tb = &tr[b * g..(b + 1) * g];
                    for c in 0..g {
                        yb[c] += xs * tb[c] as i64;
                    }
                }
            }
        }
    }

    /// out[r] (Q16) = Σ_c W[r,c] · x[c] (x en Q16). Suma de trits primero, una escala por bloque.
    fn dot_rows(&self, x: &[i32], out: &mut [i32]) {
        let (n, g) = (self.cols, self.group);
        let nb = n / g;
        for r in 0..self.rows {
            let mut acc = 0i64;
            for b in 0..nb {
                let tb = &self.t[r * n + b * g..r * n + (b + 1) * g];
                let xb = &x[b * g..(b + 1) * g];
                let mut part = 0i64;
                for c in 0..g {
                    part += xb[c] as i64 * tb[c] as i64;
                }
                acc += (part * self.s[r * nb + b] as i64) >> self.sh;
            }
            out[r] = acc as i32;
        }
    }

    /// Valor de W[r,c] en Q16.
    fn value_q16(&self, r: usize, c: usize) -> i32 {
        let v = self.t[r * self.cols + c] as i64 * self.s[r * (self.cols / self.group) + c / self.group] as i64;
        (if self.sh >= 16 { v >> (self.sh - 16) } else { v << (16 - self.sh) }) as i32
    }
}

pub struct ILayer {
    pub ln1_g: Vec<i32>,
    pub ln1_b: Vec<i32>,
    pub qkv: TMat,
    pub qkv_b: Vec<i32>,
    pub proj: TMat,
    pub proj_b: Vec<i32>,
    pub ln2_g: Vec<i32>,
    pub ln2_b: Vec<i32>,
    pub fc: TMat,
    pub fc_b: Vec<i32>,
    pub fc2: TMat,
    pub fc2_b: Vec<i32>,
}

pub struct IntModel {
    pub vocab: usize,
    pub ctx: usize,
    pub d: usize,
    pub heads: usize,
    /// 1/sqrt(dim por cabeza) en Q16.
    pub att_scale: i32,
    /// 2^(-(i+0.5)/2^EXP2_BITS) en Q30.
    pub exp2_tab: Vec<i32>,
    /// GELU(-8 + i/256) en Q16, 16·256+1 entradas.
    pub gelu_tab: Vec<i32>,
    pub wte: TMat,
    pub wpe: Vec<i32>,
    pub layers: Vec<ILayer>,
    pub lnf_g: Vec<i32>,
    pub lnf_b: Vec<i32>,
}

pub fn isqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

fn layernorm(x: &[i32], g: &[i32], b: &[i32], t: usize, d: usize) -> Vec<i32> {
    let mut out = vec![0i32; t * d];
    for i in 0..t {
        let xr = &x[i * d..(i + 1) * d];
        let mean = xr.iter().map(|&v| v as i64).sum::<i64>() / d as i64;
        let var = xr.iter().map(|&v| (v as i64 - mean) * (v as i64 - mean)).sum::<i64>() / d as i64; // Q32
        let std = isqrt((var + LN_EPS_Q32) as u64) as i64; // Q16
        let std = std.max(1);
        for j in 0..d {
            let xhat = ((xr[j] as i64 - mean) << Q) / std; // Q16
            out[i * d + j] = ((xhat * g[j] as i64) >> Q) as i32 + b[j];
        }
    }
    out
}

impl IntModel {
    /// e^(-z) con z ≥ 0 en Q16; resultado en Q30.
    fn exp_neg(&self, z: i64) -> i64 {
        let y = (z * LOG2E_Q16) >> Q; // log2 en Q16
        let ip = y >> Q;
        if ip >= 31 {
            return 0;
        }
        let fp = (y & ((1 << Q) - 1)) as usize;
        (self.exp2_tab[fp >> (Q - EXP2_BITS)] as i64) >> ip
    }

    fn gelu(&self, x: i32) -> i32 {
        let lim = GELU_RANGE << Q;
        let xv = x as i64;
        if xv >= lim {
            return x;
        }
        if xv < -lim {
            return 0;
        }
        let u = xv + lim;
        let shift = Q - GELU_STEP_BITS;
        let idx = (u >> shift) as usize;
        let frac = u & ((1 << shift) - 1);
        let a = self.gelu_tab[idx] as i64;
        let b = self.gelu_tab[idx + 1] as i64;
        (a + (((b - a) * frac) >> shift)) as i32
    }

    fn attention(&self, qkv: &[i32], t: usize) -> Vec<i32> {
        let (d, nh) = (self.d, self.heads);
        let hd = d / nh;
        let mut ao = vec![0i32; t * d];
        let mut sc = vec![0i64; t];
        let mut e = vec![0i64; t];
        let mut acc = vec![0i64; hd];
        for h in 0..nh {
            for i in 0..t {
                let q = &qkv[i * 3 * d + h * hd..][..hd];
                let mut mx = i64::MIN;
                for j in 0..=i {
                    let k = &qkv[j * 3 * d + d + h * hd..][..hd];
                    let mut dot = 0i64;
                    for c in 0..hd {
                        dot += q[c] as i64 * k[c] as i64; // Q32
                    }
                    let s = ((dot >> Q) * self.att_scale as i64) >> Q; // Q16
                    sc[j] = s;
                    mx = mx.max(s);
                }
                let mut sum = 0i64;
                for j in 0..=i {
                    e[j] = self.exp_neg(mx - sc[j]);
                    sum += e[j];
                }
                acc.fill(0);
                for j in 0..=i {
                    let p = (e[j] << Q) / sum; // Q16
                    if p == 0 {
                        continue;
                    }
                    let v = &qkv[j * 3 * d + 2 * d + h * hd..][..hd];
                    for c in 0..hd {
                        acc[c] += p * v[c] as i64; // Q32
                    }
                }
                for c in 0..hd {
                    ao[i * d + h * hd + c] = (acc[c] >> Q) as i32;
                }
            }
        }
        ao
    }

    fn linear(&self, w: &TMat, b: &[i32], x: &[i32], t: usize) -> Vec<i32> {
        let n = w.cols;
        let mut acc = vec![0i64; t * n];
        w.mul_acc(&mut acc, x, t);
        let mut out = vec![0i32; t * n];
        for i in 0..t {
            for j in 0..n {
                out[i * n + j] = (acc[i * n + j] >> Q) as i32 + b[j];
            }
        }
        out
    }

    /// Logits en Q16 para cada posición (t × vocab).
    pub fn forward(&self, inp: &[u8]) -> Vec<i32> {
        let (t, d, v) = (inp.len(), self.d, self.vocab);
        assert!(t >= 1 && t <= self.ctx);
        let mut x = vec![0i32; t * d];
        for i in 0..t {
            for j in 0..d {
                x[i * d + j] = self.wte.value_q16(inp[i] as usize, j) + self.wpe[i * d + j];
            }
        }
        for l in &self.layers {
            let ln1 = layernorm(&x, &l.ln1_g, &l.ln1_b, t, d);
            let qkv = self.linear(&l.qkv, &l.qkv_b, &ln1, t);
            let ao = self.attention(&qkv, t);
            let pr = self.linear(&l.proj, &l.proj_b, &ao, t);
            for (a, b) in x.iter_mut().zip(&pr) {
                *a += b;
            }
            let ln2 = layernorm(&x, &l.ln2_g, &l.ln2_b, t, d);
            let mut h = self.linear(&l.fc, &l.fc_b, &ln2, t);
            for hv in h.iter_mut() {
                *hv = self.gelu(*hv);
            }
            let o = self.linear(&l.fc2, &l.fc2_b, &h, t);
            for (a, b) in x.iter_mut().zip(&o) {
                *a += b;
            }
        }
        let lnf = layernorm(&x, &self.lnf_g, &self.lnf_b, t, d);
        let mut logits = vec![0i32; t * v];
        for i in 0..t {
            self.wte.dot_rows(&lnf[i * d..(i + 1) * d], &mut logits[i * v..(i + 1) * v]);
        }
        logits
    }

    /// Muestrea con temperatura num/den usando sólo enteros.
    pub fn sample(&self, logits: &[i32], temp_num: i64, temp_den: i64, rng: &mut u64) -> u8 {
        let mx = *logits.iter().max().unwrap() as i64;
        let w: Vec<i64> = logits
            .iter()
            .map(|&l| self.exp_neg((mx - l as i64) * temp_den / temp_num))
            .collect();
        let total: i64 = w.iter().sum();
        let mut r = (xorshift(rng) % total as u64) as i64;
        for (i, wi) in w.iter().enumerate() {
            r -= wi;
            if r < 0 {
                return i as u8;
            }
        }
        (logits.len() - 1) as u8
    }
}

pub fn xorshift(s: &mut u64) -> u64 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    x
}

// ---------- archivo .lli (sólo enteros) ----------

const MAGIC: &[u8; 4] = b"LLI1";

fn w_u32(o: &mut Vec<u8>, v: u32) {
    o.extend_from_slice(&v.to_le_bytes());
}
fn w_i32s(o: &mut Vec<u8>, v: &[i32]) {
    w_u32(o, v.len() as u32);
    for x in v {
        o.extend_from_slice(&x.to_le_bytes());
    }
}
fn w_tmat(o: &mut Vec<u8>, m: &TMat) {
    for v in [m.rows as u32, m.cols as u32, m.group as u32, m.sh as u32] {
        w_u32(o, v);
    }
    // 4 trits por byte: 0 → 0, 1 → +1, 2 → -1
    for ch in m.t.chunks(4) {
        let mut b = 0u8;
        for (i, &t) in ch.iter().enumerate() {
            let code = match t {
                1 => 1u8,
                -1 => 2u8,
                _ => 0u8,
            };
            b |= code << (2 * i);
        }
        o.push(b);
    }
    for s in &m.s {
        o.extend_from_slice(&s.to_le_bytes());
    }
}

struct Rd<'a>(&'a [u8], usize);
impl Rd<'_> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.0[self.1..self.1 + 4].try_into().unwrap());
        self.1 += 4;
        v
    }
    fn i32s(&mut self) -> Vec<i32> {
        let n = self.u32() as usize;
        (0..n).map(|_| self.u32() as i32).collect()
    }
    fn tmat(&mut self) -> TMat {
        let (rows, cols, group, sh) = (self.u32() as usize, self.u32() as usize, self.u32() as usize, self.u32() as i32);
        let n = rows * cols;
        let nbytes = n.div_ceil(4);
        let mut t = Vec::with_capacity(n);
        for i in 0..n {
            let code = (self.0[self.1 + i / 4] >> (2 * (i % 4))) & 3;
            t.push(match code {
                1 => 1i8,
                2 => -1i8,
                _ => 0i8,
            });
        }
        self.1 += nbytes;
        let ns = rows * (cols / group);
        let s = (0..ns)
            .map(|i| i16::from_le_bytes(self.0[self.1 + 2 * i..self.1 + 2 * i + 2].try_into().unwrap()))
            .collect();
        self.1 += 2 * ns;
        TMat { rows, cols, group, sh, t, s }
    }
}

impl IntModel {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut o = Vec::new();
        o.extend_from_slice(MAGIC);
        for v in [self.vocab, self.ctx, self.d, self.heads, self.layers.len()] {
            w_u32(&mut o, v as u32);
        }
        w_u32(&mut o, self.att_scale as u32);
        w_i32s(&mut o, &self.exp2_tab);
        w_i32s(&mut o, &self.gelu_tab);
        w_tmat(&mut o, &self.wte);
        w_i32s(&mut o, &self.wpe);
        for l in &self.layers {
            w_i32s(&mut o, &l.ln1_g);
            w_i32s(&mut o, &l.ln1_b);
            w_tmat(&mut o, &l.qkv);
            w_i32s(&mut o, &l.qkv_b);
            w_tmat(&mut o, &l.proj);
            w_i32s(&mut o, &l.proj_b);
            w_i32s(&mut o, &l.ln2_g);
            w_i32s(&mut o, &l.ln2_b);
            w_tmat(&mut o, &l.fc);
            w_i32s(&mut o, &l.fc_b);
            w_tmat(&mut o, &l.fc2);
            w_i32s(&mut o, &l.fc2_b);
        }
        w_i32s(&mut o, &self.lnf_g);
        w_i32s(&mut o, &self.lnf_b);
        o
    }

    pub fn from_bytes(b: &[u8]) -> IntModel {
        assert_eq!(&b[..4], MAGIC, "archivo .lli inválido");
        let mut r = Rd(b, 4);
        let (vocab, ctx, d, heads, nl) = (r.u32() as usize, r.u32() as usize, r.u32() as usize, r.u32() as usize, r.u32() as usize);
        let att_scale = r.u32() as i32;
        let exp2_tab = r.i32s();
        let gelu_tab = r.i32s();
        let wte = r.tmat();
        let wpe = r.i32s();
        let layers = (0..nl)
            .map(|_| ILayer {
                ln1_g: r.i32s(),
                ln1_b: r.i32s(),
                qkv: r.tmat(),
                qkv_b: r.i32s(),
                proj: r.tmat(),
                proj_b: r.i32s(),
                ln2_g: r.i32s(),
                ln2_b: r.i32s(),
                fc: r.tmat(),
                fc_b: r.i32s(),
                fc2: r.tmat(),
                fc2_b: r.i32s(),
            })
            .collect();
        let lnf_g = r.i32s();
        let lnf_b = r.i32s();
        IntModel { vocab, ctx, d, heads, att_scale, exp2_tab, gelu_tab, wte, wpe, layers, lnf_g, lnf_b }
    }

    pub fn save(&self, path: &str) {
        std::fs::File::create(path).unwrap().write_all(&self.to_bytes()).unwrap();
    }

    pub fn load(path: &str) -> IntModel {
        let mut b = Vec::new();
        std::fs::File::open(path).unwrap_or_else(|e| panic!("no pude abrir {path}: {e}")).read_to_end(&mut b).unwrap();
        IntModel::from_bytes(&b)
    }
}
