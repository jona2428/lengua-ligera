//! Exporta un modelo entrenado con QAT (ternario/binario) al motor entero.
//! La coma flotante se usa sólo aquí, una vez, para fundir el molde.
use crate::intinfer::{IntModel, ILayer, TMat, EXP2_BITS, GELU_RANGE, GELU_STEP_BITS, Q};
use crate::model::Model;
use crate::ops::gelu;
use crate::quant::Fmt;

fn q16(v: &[f32]) -> Vec<i32> {
    v.iter().map(|x| (*x as f64 * (1u64 << Q) as f64).round() as i32).collect()
}

fn tmat(w: &[f32], rows: usize, cols: usize, group: usize) -> TMat {
    assert!(cols % group == 0);
    let mut t = Vec::with_capacity(rows * cols);
    let mut sc = Vec::with_capacity(rows * cols / group);
    for blk in w.chunks(group) {
        let s = blk.iter().fold(0f32, |m, x| m.max(x.abs()));
        for x in blk {
            // el cuantizador deja cada valor en {-s, 0, +s}
            debug_assert!(*x == 0.0 || (x.abs() - s).abs() <= 1e-6 * s.max(1e-12));
            t.push(if *x > 0.0 { 1i8 } else if *x < 0.0 { -1i8 } else { 0i8 });
        }
        sc.push(s);
    }
    let smax = sc.iter().fold(0f32, |m, x| m.max(*x)).max(1e-12);
    // mayor exponente tal que smax · 2^sh cabe en i16
    let sh = ((32767.0 / smax as f64).log2().floor() as i32).min(40);
    let s = sc.iter().map(|x| (*x as f64 * 2f64.powi(sh)).round() as i16).collect();
    TMat { rows, cols, group, sh, t, s }
}

pub fn export(m: &Model, f: &Fmt) -> IntModel {
    let group = match f {
        Fmt::Ternary { group } | Fmt::Binary { group } => *group,
        _ => panic!("el motor entero sólo acepta formatos ternario/binario"),
    };
    let c = m.cfg;
    let d = c.d;
    let mut p = m.p.clone();
    for mat in m.lay.matrices(&c) {
        f.apply(&mut p[mat.off..mat.off + mat.rows * mat.cols]);
    }
    let sl = |o: usize, n: usize| &p[o..o + n];
    let hd = (d / c.heads) as f64;
    let exp2_tab = (0..1usize << EXP2_BITS)
        .map(|i| (2f64.powf(-(i as f64 + 0.5) / (1u64 << EXP2_BITS) as f64) * (1u64 << 30) as f64).round() as i32)
        .collect();
    let steps = (2 * GELU_RANGE as usize) << GELU_STEP_BITS;
    let gelu_tab = (0..=steps)
        .map(|i| {
            let x = -(GELU_RANGE as f64) + i as f64 / (1u64 << GELU_STEP_BITS) as f64;
            (gelu(x as f32) as f64 * (1u64 << Q) as f64).round() as i32
        })
        .collect();
    let layers = m
        .lay
        .layers
        .iter()
        .map(|l| ILayer {
            ln1_g: q16(sl(l.ln1_g, d)),
            ln1_b: q16(sl(l.ln1_g + d, d)),
            qkv: tmat(sl(l.qkv_w, 3 * d * d), d, 3 * d, group),
            qkv_b: q16(sl(l.qkv_b, 3 * d)),
            proj: tmat(sl(l.proj_w, d * d), d, d, group),
            proj_b: q16(sl(l.proj_b, d)),
            ln2_g: q16(sl(l.ln2_g, d)),
            ln2_b: q16(sl(l.ln2_g + d, d)),
            fc: tmat(sl(l.fc_w, 4 * d * d), d, 4 * d, group),
            fc_b: q16(sl(l.fc_b, 4 * d)),
            fc2: tmat(sl(l.fc2_w, 4 * d * d), 4 * d, d, group),
            fc2_b: q16(sl(l.fc2_b, d)),
        })
        .collect();
    IntModel {
        vocab: c.vocab,
        ctx: c.ctx,
        d,
        heads: c.heads,
        att_scale: ((1u64 << Q) as f64 / hd.sqrt()).round() as i32,
        exp2_tab,
        gelu_tab,
        wte: tmat(sl(m.lay.wte, c.vocab * d), c.vocab, d, group),
        wpe: q16(sl(m.lay.wpe, c.ctx * d)),
        layers,
        lnf_g: q16(sl(m.lay.lnf_g, d)),
        lnf_b: q16(sl(m.lay.lnf_g + d, d)),
        act_bits: 0,
    }
}
