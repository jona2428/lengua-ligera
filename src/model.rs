//! Transformer de caracteres (pre-LN, GELU, embeddings atados) con backprop a mano.
//! Todos los parámetros viven en un solo Vec<f32>; `Layout` guarda los offsets.
use crate::ops::{dot, gelu, gelu_grad, mm_acc, transpose};
use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub vocab: usize,
    pub ctx: usize,
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
}

#[derive(Clone, Debug)]
pub struct LayerOff {
    pub ln1_g: usize, // ln1_b = ln1_g + d
    pub qkv_w: usize,
    pub qkv_b: usize,
    pub proj_w: usize,
    pub proj_b: usize,
    pub ln2_g: usize, // ln2_b = ln2_g + d
    pub fc_w: usize,
    pub fc_b: usize,
    pub fc2_w: usize,
    pub fc2_b: usize,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub wte: usize,
    pub wpe: usize,
    pub layers: Vec<LayerOff>,
    pub lnf_g: usize, // lnf_b = lnf_g + d
    pub total: usize,
}

/// Una matriz de pesos (candidata a cuantizar).
#[derive(Clone, Debug)]
pub struct Matrix {
    pub name: String,
    pub off: usize,
    pub rows: usize,
    pub cols: usize,
}

impl Layout {
    pub fn new(c: &Config) -> Self {
        let d = c.d;
        let mut off = 0usize;
        let mut take = |n: usize| {
            let o = off;
            off += n;
            o
        };
        let wte = take(c.vocab * d);
        let wpe = take(c.ctx * d);
        let mut layers = Vec::new();
        for _ in 0..c.layers {
            let ln1_g = take(2 * d);
            let qkv_w = take(d * 3 * d);
            let qkv_b = take(3 * d);
            let proj_w = take(d * d);
            let proj_b = take(d);
            let ln2_g = take(2 * d);
            let fc_w = take(d * 4 * d);
            let fc_b = take(4 * d);
            let fc2_w = take(4 * d * d);
            let fc2_b = take(d);
            layers.push(LayerOff { ln1_g, qkv_w, qkv_b, proj_w, proj_b, ln2_g, fc_w, fc_b, fc2_w, fc2_b });
        }
        let lnf_g = take(2 * d);
        let total = take(0);
        Layout { wte, wpe, layers, lnf_g, total }
    }

    pub fn matrices(&self, c: &Config) -> Vec<Matrix> {
        let d = c.d;
        let m = |name: String, off, rows, cols| Matrix { name, off, rows, cols };
        let mut v = vec![m("wte".into(), self.wte, c.vocab, d)];
        for (i, l) in self.layers.iter().enumerate() {
            v.push(m(format!("l{i}.qkv"), l.qkv_w, d, 3 * d));
            v.push(m(format!("l{i}.proj"), l.proj_w, d, d));
            v.push(m(format!("l{i}.fc"), l.fc_w, d, 4 * d));
            v.push(m(format!("l{i}.fc2"), l.fc2_w, 4 * d, d));
        }
        v
    }
}

pub struct Model {
    pub cfg: Config,
    pub lay: Layout,
    pub p: Vec<f32>,
}

/// Copias transpuestas de las matrices, recalculadas una vez por paso.
pub struct Wt {
    pub wte_t: Vec<f32>,
    pub qkv_t: Vec<Vec<f32>>,
    pub proj_t: Vec<Vec<f32>>,
    pub fc_t: Vec<Vec<f32>>,
    pub fc2_t: Vec<Vec<f32>>,
}

pub struct LayerActs {
    pub ln1: Vec<f32>,
    pub ln1_mu: Vec<f32>,
    pub ln1_rs: Vec<f32>,
    pub qkv: Vec<f32>,
    pub att: Vec<f32>,
    pub ao: Vec<f32>,
    pub res1: Vec<f32>,
    pub ln2: Vec<f32>,
    pub ln2_mu: Vec<f32>,
    pub ln2_rs: Vec<f32>,
    pub h: Vec<f32>,
    pub hg: Vec<f32>,
    pub res2: Vec<f32>,
}

pub struct Acts {
    pub x0: Vec<f32>,
    pub layers: Vec<LayerActs>,
    pub lnf: Vec<f32>,
    pub lnf_mu: Vec<f32>,
    pub lnf_rs: Vec<f32>,
    /// Probabilidades tras forward; gradiente de logits durante backward.
    pub probs: Vec<f32>,
    dx: Vec<f32>,
    dtmp: Vec<f32>,
    dbig: Vec<f32>,
    dqkv: Vec<f32>,
    dp: Vec<f32>,
}

impl Acts {
    pub fn new(c: &Config) -> Self {
        let (t, d) = (c.ctx, c.d);
        let z = |n: usize| vec![0f32; n];
        let layers = (0..c.layers)
            .map(|_| LayerActs {
                ln1: z(t * d),
                ln1_mu: z(t),
                ln1_rs: z(t),
                qkv: z(t * 3 * d),
                att: z(c.heads * t * t),
                ao: z(t * d),
                res1: z(t * d),
                ln2: z(t * d),
                ln2_mu: z(t),
                ln2_rs: z(t),
                h: z(t * 4 * d),
                hg: z(t * 4 * d),
                res2: z(t * d),
            })
            .collect();
        Acts {
            x0: z(t * d),
            layers,
            lnf: z(t * d),
            lnf_mu: z(t),
            lnf_rs: z(t),
            probs: z(t * c.vocab),
            dx: z(t * d),
            dtmp: z(t * d),
            dbig: z(t * 4 * d),
            dqkv: z(t * 3 * d),
            dp: z(t),
        }
    }
}

fn layernorm(out: &mut [f32], mu: &mut [f32], rs: &mut [f32], x: &[f32], gb: &[f32], t: usize, d: usize) {
    let (g, b) = gb.split_at(d);
    for i in 0..t {
        let xr = &x[i * d..(i + 1) * d];
        let m = xr.iter().sum::<f32>() / d as f32;
        let v = xr.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / d as f32;
        let r = 1.0 / (v + 1e-5).sqrt();
        mu[i] = m;
        rs[i] = r;
        let o = &mut out[i * d..(i + 1) * d];
        for j in 0..d {
            o[j] = (xr[j] - m) * r * g[j] + b[j];
        }
    }
}

/// Acumula en dx; dgb = [dg | db].
#[allow(clippy::too_many_arguments)]
fn layernorm_back(dx: &mut [f32], dgb: &mut [f32], dy: &[f32], x: &[f32], mu: &[f32], rs: &[f32], g: &[f32], t: usize, d: usize) {
    let (dg, db) = dgb.split_at_mut(d);
    for i in 0..t {
        let (m, r) = (mu[i], rs[i]);
        let (xr, dyr) = (&x[i * d..(i + 1) * d], &dy[i * d..(i + 1) * d]);
        let (mut s1, mut s2) = (0f32, 0f32);
        for j in 0..d {
            let xh = (xr[j] - m) * r;
            let dxh = dyr[j] * g[j];
            s1 += dxh;
            s2 += dxh * xh;
            dg[j] += dyr[j] * xh;
            db[j] += dyr[j];
        }
        s1 /= d as f32;
        s2 /= d as f32;
        let dxr = &mut dx[i * d..(i + 1) * d];
        for j in 0..d {
            let xh = (xr[j] - m) * r;
            dxr[j] += r * (dyr[j] * g[j] - s1 - xh * s2);
        }
    }
}

fn set_rows(y: &mut [f32], b: &[f32], t: usize) {
    let n = b.len();
    for i in 0..t {
        y[i * n..(i + 1) * n].copy_from_slice(b);
    }
}

fn add_col_sums(db: &mut [f32], dy: &[f32], t: usize) {
    let n = db.len();
    for i in 0..t {
        for (a, b) in db.iter_mut().zip(&dy[i * n..(i + 1) * n]) {
            *a += b;
        }
    }
}

fn attn_fwd(ao: &mut [f32], att: &mut [f32], qkv: &[f32], t: usize, d: usize, nh: usize) {
    let hd = d / nh;
    let sc = 1.0 / (hd as f32).sqrt();
    for h in 0..nh {
        for i in 0..t {
            let q = &qkv[i * 3 * d + h * hd..][..hd];
            let row = &mut att[(h * t + i) * t..(h * t + i + 1) * t];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..=i {
                let s = dot(q, &qkv[j * 3 * d + d + h * hd..][..hd]) * sc;
                row[j] = s;
                mx = mx.max(s);
            }
            let mut z = 0f32;
            for r in row[..=i].iter_mut() {
                *r = (*r - mx).exp();
                z += *r;
            }
            for r in row[..=i].iter_mut() {
                *r /= z;
            }
            row[i + 1..].fill(0.0);
            let o = &mut ao[i * d + h * hd..][..hd];
            o.fill(0.0);
            for j in 0..=i {
                let p = row[j];
                let v = &qkv[j * 3 * d + 2 * d + h * hd..][..hd];
                for c in 0..hd {
                    o[c] += p * v[c];
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn attn_bwd(dqkv: &mut [f32], dao: &[f32], qkv: &[f32], att: &[f32], dp: &mut [f32], t: usize, d: usize, nh: usize) {
    let hd = d / nh;
    let sc = 1.0 / (hd as f32).sqrt();
    for h in 0..nh {
        for i in 0..t {
            let row = &att[(h * t + i) * t..(h * t + i + 1) * t];
            let g = &dao[i * d + h * hd..][..hd];
            let mut s = 0f32;
            for j in 0..=i {
                let v = &qkv[j * 3 * d + 2 * d + h * hd..][..hd];
                dp[j] = dot(g, v);
                s += row[j] * dp[j];
                let dv = &mut dqkv[j * 3 * d + 2 * d + h * hd..][..hd];
                for c in 0..hd {
                    dv[c] += row[j] * g[c];
                }
            }
            for j in 0..=i {
                let ds = row[j] * (dp[j] - s) * sc;
                if ds == 0.0 {
                    continue;
                }
                for c in 0..hd {
                    dqkv[i * 3 * d + h * hd + c] += ds * qkv[j * 3 * d + d + h * hd + c];
                    dqkv[j * 3 * d + d + h * hd + c] += ds * qkv[i * 3 * d + h * hd + c];
                }
            }
        }
    }
}

impl Model {
    pub fn zeros(cfg: Config) -> Self {
        assert!(cfg.d % 16 == 0 && cfg.vocab % 16 == 0 && cfg.d % cfg.heads == 0);
        let lay = Layout::new(&cfg);
        let p = vec![0f32; lay.total];
        Model { cfg, lay, p }
    }

    pub fn init(cfg: Config, seed: u64) -> Self {
        let mut m = Model::zeros(cfg);
        let mut rng = Rng::new(seed);
        let d = cfg.d;
        let proj_std = 0.02 / (2.0 * cfg.layers as f32).sqrt();
        let mut fill = |p: &mut [f32], std: f32| p.iter_mut().for_each(|x| *x = rng.normal() * std);
        let lay = m.lay.clone();
        fill(&mut m.p[lay.wte..lay.wte + cfg.vocab * d], 0.02);
        fill(&mut m.p[lay.wpe..lay.wpe + cfg.ctx * d], 0.01);
        for l in &lay.layers {
            m.p[l.ln1_g..l.ln1_g + d].fill(1.0);
            m.p[l.ln2_g..l.ln2_g + d].fill(1.0);
            fill(&mut m.p[l.qkv_w..l.qkv_w + 3 * d * d], 0.02);
            fill(&mut m.p[l.proj_w..l.proj_w + d * d], proj_std);
            fill(&mut m.p[l.fc_w..l.fc_w + 4 * d * d], 0.02);
            fill(&mut m.p[l.fc2_w..l.fc2_w + 4 * d * d], proj_std);
        }
        m.p[lay.lnf_g..lay.lnf_g + d].fill(1.0);
        m
    }

    pub fn n_params(&self) -> usize {
        self.p.len()
    }

    pub fn transposed(&self) -> Wt {
        let (c, p, d) = (&self.cfg, &self.p, self.cfg.d);
        let tr = |off: usize, r: usize, cc: usize| transpose(&p[off..off + r * cc], r, cc);
        Wt {
            wte_t: tr(self.lay.wte, c.vocab, d),
            qkv_t: self.lay.layers.iter().map(|l| tr(l.qkv_w, d, 3 * d)).collect(),
            proj_t: self.lay.layers.iter().map(|l| tr(l.proj_w, d, d)).collect(),
            fc_t: self.lay.layers.iter().map(|l| tr(l.fc_w, d, 4 * d)).collect(),
            fc2_t: self.lay.layers.iter().map(|l| tr(l.fc2_w, 4 * d, d)).collect(),
        }
    }

    /// Forward sobre `inp` (len ≤ ctx). Deja probabilidades en `a.probs`.
    /// Devuelve la pérdida media (nats) si hay objetivos.
    pub fn forward(&self, wt: &Wt, a: &mut Acts, inp: &[u8], tgt: Option<&[u8]>) -> f32 {
        let c = self.cfg;
        let (t, d, v, nh) = (inp.len(), c.d, c.vocab, c.heads);
        assert!(t >= 1 && t <= c.ctx);
        let (p, lay) = (&self.p, &self.lay);
        for i in 0..t {
            let e = &p[lay.wte + inp[i] as usize * d..][..d];
            let pe = &p[lay.wpe + i * d..][..d];
            for j in 0..d {
                a.x0[i * d + j] = e[j] + pe[j];
            }
        }
        for l in 0..c.layers {
            let o = &lay.layers[l];
            let (prev, rest) = a.layers.split_at_mut(l);
            let la = &mut rest[0];
            let xin: &[f32] = if l == 0 { &a.x0[..t * d] } else { &prev[l - 1].res2[..t * d] };
            layernorm(&mut la.ln1, &mut la.ln1_mu, &mut la.ln1_rs, xin, &p[o.ln1_g..o.ln1_g + 2 * d], t, d);
            set_rows(&mut la.qkv, &p[o.qkv_b..o.qkv_b + 3 * d], t);
            mm_acc(&mut la.qkv, &la.ln1, &p[o.qkv_w..], t, d, 3 * d);
            attn_fwd(&mut la.ao, &mut la.att, &la.qkv, t, d, nh);
            la.res1[..t * d].copy_from_slice(xin);
            for i in 0..t {
                for j in 0..d {
                    la.res1[i * d + j] += p[o.proj_b + j];
                }
            }
            mm_acc(&mut la.res1, &la.ao, &p[o.proj_w..], t, d, d);
            layernorm(&mut la.ln2, &mut la.ln2_mu, &mut la.ln2_rs, &la.res1[..t * d], &p[o.ln2_g..o.ln2_g + 2 * d], t, d);
            set_rows(&mut la.h, &p[o.fc_b..o.fc_b + 4 * d], t);
            mm_acc(&mut la.h, &la.ln2, &p[o.fc_w..], t, d, 4 * d);
            for i in 0..t * 4 * d {
                la.hg[i] = gelu(la.h[i]);
            }
            la.res2[..t * d].copy_from_slice(&la.res1[..t * d]);
            for i in 0..t {
                for j in 0..d {
                    la.res2[i * d + j] += p[o.fc2_b + j];
                }
            }
            mm_acc(&mut la.res2, &la.hg, &p[o.fc2_w..], t, 4 * d, d);
        }
        let xl: &[f32] = if c.layers == 0 { &a.x0[..t * d] } else { &a.layers[c.layers - 1].res2[..t * d] };
        layernorm(&mut a.lnf, &mut a.lnf_mu, &mut a.lnf_rs, xl, &p[lay.lnf_g..lay.lnf_g + 2 * d], t, d);
        a.probs[..t * v].fill(0.0);
        mm_acc(&mut a.probs, &a.lnf, &wt.wte_t, t, d, v);
        let mut loss = 0f64;
        for i in 0..t {
            let r = &mut a.probs[i * v..(i + 1) * v];
            let mx = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut z = 0f32;
            for x in r.iter_mut() {
                *x = (*x - mx).exp();
                z += *x;
            }
            for x in r.iter_mut() {
                *x /= z;
            }
            if let Some(tg) = tgt {
                loss -= (r[tg[i] as usize].max(1e-30) as f64).ln();
            }
        }
        (loss / t as f64) as f32
    }

    /// Backward tras `forward` con los mismos datos. Acumula en `g` escalado por `gscale`
    /// (gradiente de gscale · Σ_t pérdida_t).
    pub fn backward(&self, wt: &Wt, a: &mut Acts, inp: &[u8], tgt: &[u8], g: &mut [f32], gscale: f32) {
        let c = self.cfg;
        let (t, d, v, nh) = (inp.len(), c.d, c.vocab, c.heads);
        let (p, lay) = (&self.p, &self.lay);
        for i in 0..t {
            let r = &mut a.probs[i * v..(i + 1) * v];
            r[tgt[i] as usize] -= 1.0;
            for x in r.iter_mut() {
                *x *= gscale;
            }
        }
        let dl_t = transpose(&a.probs[..t * v], t, v);
        mm_acc(&mut g[lay.wte..], &dl_t, &a.lnf, v, t, d);
        a.dtmp[..t * d].fill(0.0);
        mm_acc(&mut a.dtmp, &a.probs, &p[lay.wte..], t, v, d);
        a.dx[..t * d].fill(0.0);
        {
            let xl: &[f32] = if c.layers == 0 { &a.x0 } else { &a.layers[c.layers - 1].res2 };
            let lg = lay.lnf_g;
            layernorm_back(&mut a.dx, &mut g[lg..lg + 2 * d], &a.dtmp, xl, &a.lnf_mu, &a.lnf_rs, &p[lg..lg + d], t, d);
        }
        for l in (0..c.layers).rev() {
            let o = &lay.layers[l];
            let la = &a.layers[l];
            let xin: &[f32] = if l == 0 { &a.x0 } else { &a.layers[l - 1].res2 };
            // MLP
            add_col_sums(&mut g[o.fc2_b..o.fc2_b + d], &a.dx, t);
            let hg_t = transpose(&la.hg[..t * 4 * d], t, 4 * d);
            mm_acc(&mut g[o.fc2_w..], &hg_t, &a.dx, 4 * d, t, d);
            a.dbig[..t * 4 * d].fill(0.0);
            mm_acc(&mut a.dbig, &a.dx, &wt.fc2_t[l], t, d, 4 * d);
            for i in 0..t * 4 * d {
                a.dbig[i] *= gelu_grad(la.h[i]);
            }
            add_col_sums(&mut g[o.fc_b..o.fc_b + 4 * d], &a.dbig, t);
            let ln2_t = transpose(&la.ln2[..t * d], t, d);
            mm_acc(&mut g[o.fc_w..], &ln2_t, &a.dbig, d, t, 4 * d);
            a.dtmp[..t * d].fill(0.0);
            mm_acc(&mut a.dtmp, &a.dbig, &wt.fc_t[l], t, 4 * d, d);
            layernorm_back(&mut a.dx, &mut g[o.ln2_g..o.ln2_g + 2 * d], &a.dtmp, &la.res1, &la.ln2_mu, &la.ln2_rs, &p[o.ln2_g..o.ln2_g + d], t, d);
            // atención
            add_col_sums(&mut g[o.proj_b..o.proj_b + d], &a.dx, t);
            let ao_t = transpose(&la.ao[..t * d], t, d);
            mm_acc(&mut g[o.proj_w..], &ao_t, &a.dx, d, t, d);
            a.dtmp[..t * d].fill(0.0);
            mm_acc(&mut a.dtmp, &a.dx, &wt.proj_t[l], t, d, d);
            a.dqkv[..t * 3 * d].fill(0.0);
            attn_bwd(&mut a.dqkv, &a.dtmp, &la.qkv, &la.att, &mut a.dp, t, d, nh);
            add_col_sums(&mut g[o.qkv_b..o.qkv_b + 3 * d], &a.dqkv, t);
            let ln1_t = transpose(&la.ln1[..t * d], t, d);
            mm_acc(&mut g[o.qkv_w..], &ln1_t, &a.dqkv, d, t, 3 * d);
            a.dtmp[..t * d].fill(0.0);
            mm_acc(&mut a.dtmp, &a.dqkv, &wt.qkv_t[l], t, 3 * d, d);
            layernorm_back(&mut a.dx, &mut g[o.ln1_g..o.ln1_g + 2 * d], &a.dtmp, xin, &la.ln1_mu, &la.ln1_rs, &p[o.ln1_g..o.ln1_g + d], t, d);
        }
        for i in 0..t {
            let tok = inp[i] as usize;
            for j in 0..d {
                let gx = a.dx[i * d + j];
                g[lay.wte + tok * d + j] += gx;
                g[lay.wpe + i * d + j] += gx;
            }
        }
    }
}
