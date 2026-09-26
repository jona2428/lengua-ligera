//! Velocidad y memoria de generación, de a un carácter con caché KV, en un hilo.
//! Compara el motor float (misma red, pesos ternarizados en f32) con el motor entero
//! (activaciones Q16 o int8). El float es sólo la vara de medir: no forma parte del entregable.
//! Uso: MODE=int|float|check ACT_BITS=8 LLI=runs/tern32.lli MODEL=runs/qat_tern32.llg REPS=20 bench_gen
use lengua_ligera::data::{env, env_str, load_bin, load_model};
use lengua_ligera::intinfer::IntModel;
use lengua_ligera::model::Model;
use lengua_ligera::ops::{dot, gelu, mm_acc};
use lengua_ligera::quant::Fmt;
use std::time::Instant;

#[repr(C)]
#[derive(Default)]
struct MemCounters {
    cb: u32,
    page_fault_count: u32,
    peak_working_set: usize,
    working_set: usize,
    quota_peak_paged: usize,
    quota_paged: usize,
    quota_peak_nonpaged: usize,
    quota_nonpaged: usize,
    pagefile: usize,
    peak_pagefile: usize,
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn GetCurrentProcess() -> isize;
    fn K32GetProcessMemoryInfo(p: isize, c: *mut MemCounters, cb: u32) -> i32;
}

/// (pico de memoria privada, pico de working set) en bytes.
fn peak_mem() -> (usize, usize) {
    #[cfg(windows)]
    unsafe {
        let mut c = MemCounters { cb: std::mem::size_of::<MemCounters>() as u32, ..Default::default() };
        K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb);
        (c.peak_pagefile, c.peak_working_set)
    }
    #[cfg(not(windows))]
    (0, 0)
}

/// Paso float con caché KV (referencia).
struct FloatGen<'a> {
    m: &'a Model,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    len: usize,
}

fn ln_f(x: &[f32], g: &[f32]) -> Vec<f32> {
    let d = x.len();
    let mu = x.iter().sum::<f32>() / d as f32;
    let var = x.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / d as f32;
    let rs = 1.0 / (var + 1e-5).sqrt();
    (0..d).map(|j| (x[j] - mu) * rs * g[j] + g[d + j]).collect()
}

impl FloatGen<'_> {
    fn step(&mut self, tok: u8) -> Vec<f32> {
        let (c, p, lay) = (self.m.cfg, &self.m.p, &self.m.lay);
        let (d, pos, nh) = (c.d, self.len, c.heads);
        let hd = d / nh;
        let mut x: Vec<f32> = (0..d).map(|j| p[lay.wte + tok as usize * d + j] + p[lay.wpe + pos * d + j]).collect();
        let mut sc = vec![0f32; pos + 1];
        for (l, o) in lay.layers.iter().enumerate() {
            let h1 = ln_f(&x, &p[o.ln1_g..o.ln1_g + 2 * d]);
            let mut qkv = p[o.qkv_b..o.qkv_b + 3 * d].to_vec();
            mm_acc(&mut qkv, &h1, &p[o.qkv_w..], 1, d, 3 * d);
            self.k[l][pos * d..(pos + 1) * d].copy_from_slice(&qkv[d..2 * d]);
            self.v[l][pos * d..(pos + 1) * d].copy_from_slice(&qkv[2 * d..]);
            let mut ao = vec![0f32; d];
            for h in 0..nh {
                let q = &qkv[h * hd..(h + 1) * hd];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..=pos {
                    sc[j] = dot(q, &self.k[l][j * d + h * hd..][..hd]) / (hd as f32).sqrt();
                    mx = mx.max(sc[j]);
                }
                let mut z = 0f32;
                for s in sc.iter_mut() {
                    *s = (*s - mx).exp();
                    z += *s;
                }
                for j in 0..=pos {
                    let w = sc[j] / z;
                    let vr = &self.v[l][j * d + h * hd..][..hd];
                    for cc in 0..hd {
                        ao[h * hd + cc] += w * vr[cc];
                    }
                }
            }
            let mut pr = p[o.proj_b..o.proj_b + d].to_vec();
            mm_acc(&mut pr, &ao, &p[o.proj_w..], 1, d, d);
            for j in 0..d {
                x[j] += pr[j];
            }
            let h2 = ln_f(&x, &p[o.ln2_g..o.ln2_g + 2 * d]);
            let mut hh = p[o.fc_b..o.fc_b + 4 * d].to_vec();
            mm_acc(&mut hh, &h2, &p[o.fc_w..], 1, d, 4 * d);
            for v in hh.iter_mut() {
                *v = gelu(*v);
            }
            let mut o2 = p[o.fc2_b..o.fc2_b + d].to_vec();
            mm_acc(&mut o2, &hh, &p[o.fc2_w..], 1, 4 * d, d);
            for j in 0..d {
                x[j] += o2[j];
            }
        }
        self.len += 1;
        let hf = ln_f(&x, &p[lay.lnf_g..lay.lnf_g + 2 * d]);
        (0..c.vocab).map(|v| dot(&hf, &p[lay.wte + v * d..][..d])).collect()
    }
}

fn main() {
    let mode = env_str("MODE", "int");
    let reps: usize = env("REPS", 20);
    let valid = load_bin("data/valid.bin");
    let (_, ws0) = peak_mem();
    match mode.as_str() {
        "check" => {
            // step() con caché debe dar exactamente las filas de forward()
            let mut im = IntModel::load(&env_str("LLI", "runs/tern32.lli"));
            for ab in [0u32, 8] {
                im.act_bits = ab;
                let mut bad = 0;
                for w in 0..8 {
                    let inp = &valid[w * 1000..w * 1000 + im.ctx];
                    let full = im.forward(inp);
                    let mut c = im.new_cache();
                    for (i, &tk) in inp.iter().enumerate() {
                        let r = im.step(tk, &mut c);
                        bad += (r[..] != full[i * im.vocab..(i + 1) * im.vocab]) as usize;
                    }
                }
                println!("act_bits {ab}: filas distintas step vs forward = {bad} de {}", 8 * im.ctx);
            }
        }
        "int" => {
            let path = env_str("LLI", "runs/tern32.lli");
            let mut im = IntModel::load(&path);
            im.act_bits = env("ACT_BITS", 8);
            let file = std::fs::metadata(&path).unwrap().len();
            let mut wbytes = im.wpe.len() * 4 + im.exp2_tab.len() * 4 + im.gelu_tab.len() * 4;
            let tm = |m: &lengua_ligera::intinfer::TMat| m.t.len() + m.s.len() * 2;
            wbytes += tm(&im.wte);
            for l in &im.layers {
                wbytes += tm(&l.qkv) + tm(&l.proj) + tm(&l.fc) + tm(&l.fc2);
                wbytes += 4 * (l.ln1_g.len() * 4 + l.qkv_b.len() + l.proj_b.len() + l.fc_b.len() + l.fc2_b.len());
            }
            let t0 = Instant::now();
            let mut n = 0usize;
            let mut sink = 0i64;
            for r in 0..reps {
                let mut c = im.new_cache();
                for &tk in &valid[r * 997..r * 997 + im.ctx] {
                    sink += im.step(tk, &mut c)[0] as i64;
                    n += 1;
                }
            }
            let dt = t0.elapsed().as_secs_f64();
            let kv = im.layers.len() * 2 * im.ctx * im.d * 4;
            let (pp, pw) = peak_mem();
            println!(
                "entero act_bits {}: {:.0} car/s (1 hilo, caché KV, ctx {}) | archivo {:.2} MB | pesos en RAM {:.2} MB | caché KV {:.2} MB | pico privado {:.1} MB, pico WS {:.1} MB (base {:.1}) [{sink}]",
                im.act_bits, n as f64 / dt, im.ctx, file as f64 / 1e6, wbytes as f64 / 1e6, kv as f64 / 1e6, pp as f64 / 1e6, pw as f64 / 1e6, ws0 as f64 / 1e6
            );
        }
        _ => {
            let mut m = load_model(&env_str("MODEL", "runs/qat_tern32.llg"));
            let f = Fmt::parse(&env_str("FMT", "ternario-g32")).unwrap();
            for mat in m.lay.matrices(&m.cfg) {
                f.apply(&mut m.p[mat.off..mat.off + mat.rows * mat.cols]);
            }
            let (ctx, d, nl) = (m.cfg.ctx, m.cfg.d, m.cfg.layers);
            let t0 = Instant::now();
            let mut n = 0usize;
            let mut sink = 0f32;
            for r in 0..reps {
                let mut g = FloatGen { m: &m, k: vec![vec![0.0; ctx * d]; nl], v: vec![vec![0.0; ctx * d]; nl], len: 0 };
                for &tk in &valid[r * 997..r * 997 + ctx] {
                    sink += g.step(tk)[0];
                    n += 1;
                }
            }
            let dt = t0.elapsed().as_secs_f64();
            let (pp, pw) = peak_mem();
            println!(
                "float32: {:.0} car/s (1 hilo, caché KV, ctx {ctx}) | pesos en RAM {:.2} MB | caché KV {:.2} MB | pico privado {:.1} MB, pico WS {:.1} MB (base {:.1}) [{sink:.1}]",
                n as f64 / dt, m.p.len() as f64 * 4.0 / 1e6, (nl * 2 * ctx * d * 4) as f64 / 1e6, pp as f64 / 1e6, pw as f64 / 1e6, ws0 as f64 / 1e6
            );
        }
    }
}
