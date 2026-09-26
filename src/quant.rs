//! Formatos numéricos para pesos: cuantización simulada (cuantizar y reconstruir en f32)
//! más la contabilidad de bits reales por peso, incluidas las escalas.

#[derive(Clone, Debug)]
pub enum Fmt {
    F32,
    /// Minifloat sin escala con sesgo IEEE: F16 = (5,10), BF16 = (8,7).
    Float { e: u32, m: u32 },
    /// Minifloat 1+e+m bits con escala fp16 por bloque de `group` pesos.
    MiniFloat { e: u32, m: u32, group: usize },
    /// Entero simétrico de `bits` con escala fp16 por bloque (group = 0: por matriz).
    Int { bits: u32, group: usize },
    /// {-s, 0, +s} por bloque (umbral 0.7·media|w|).
    Ternary { group: usize },
    /// {-s, +s} por bloque, s = media|w| (estilo XNOR).
    Binary { group: usize },
    /// NF4 de QLoRA: cuantiles de la normal, escala absmax por bloque.
    Nf4 { group: usize },
    /// Codebook aprendido (k-means 1D) por matriz sobre valores normalizados por bloque.
    KMeans { bits: u32, group: usize },
}

/// Catálogo de formatos del banco.
pub fn catalog() -> Vec<Fmt> {
    use Fmt::*;
    let mut v = vec![F32, Float { e: 5, m: 10 }, Float { e: 8, m: 7 }];
    for g in [32usize, 128] {
        v.push(Int { bits: 8, group: g });
        v.push(MiniFloat { e: 4, m: 3, group: g });
        v.push(MiniFloat { e: 5, m: 2, group: g });
        v.push(MiniFloat { e: 3, m: 2, group: g });
        v.push(MiniFloat { e: 2, m: 3, group: g });
        v.push(Int { bits: 6, group: g });
        v.push(Int { bits: 5, group: g });
        v.push(Int { bits: 4, group: g });
        v.push(MiniFloat { e: 2, m: 1, group: g });
        v.push(MiniFloat { e: 3, m: 0, group: g });
        v.push(MiniFloat { e: 1, m: 2, group: g });
        v.push(Nf4 { group: g });
        v.push(KMeans { bits: 4, group: g });
        v.push(Int { bits: 3, group: g });
        v.push(MiniFloat { e: 1, m: 1, group: g });
        v.push(MiniFloat { e: 2, m: 0, group: g });
        v.push(KMeans { bits: 3, group: g });
        v.push(Int { bits: 2, group: g });
        v.push(KMeans { bits: 2, group: g });
        v.push(Ternary { group: g });
        v.push(Binary { group: g });
    }
    v.push(Int { bits: 8, group: 0 });
    v.push(Int { bits: 4, group: 0 });
    v
}

impl Fmt {
    /// Busca un formato del catálogo por nombre (p. ej. "ternario-g32").
    pub fn parse(name: &str) -> Option<Fmt> {
        catalog().into_iter().find(|f| f.name() == name)
    }
}

const SCALE_BITS: f64 = 16.0;

impl Fmt {
    pub fn name(&self) -> String {
        match self {
            Fmt::F32 => "fp32".into(),
            Fmt::Float { e: 5, m: 10 } => "fp16".into(),
            Fmt::Float { e: 8, m: 7 } => "bf16".into(),
            Fmt::Float { e, m } => format!("e{e}m{m}"),
            Fmt::MiniFloat { e, m, group } => format!("fp{}-e{e}m{m}-g{group}", 1 + e + m),
            Fmt::Int { bits, group: 0 } => format!("int{bits}-tensor"),
            Fmt::Int { bits, group } => format!("int{bits}-g{group}"),
            Fmt::Ternary { group } => format!("ternario-g{group}"),
            Fmt::Binary { group } => format!("binario-g{group}"),
            Fmt::Nf4 { group } => format!("nf4-g{group}"),
            Fmt::KMeans { bits, group } => format!("kmeans{bits}-g{group}"),
        }
    }

    /// Bits por peso, incluyendo la escala por bloque.
    pub fn bits(&self) -> f64 {
        let sc = |g: &usize| if *g == 0 { 0.0 } else { SCALE_BITS / *g as f64 };
        match self {
            Fmt::F32 => 32.0,
            Fmt::Float { e, m } => (1 + e + m) as f64,
            Fmt::MiniFloat { e, m, group } => (1 + e + m) as f64 + sc(group),
            Fmt::Int { bits, group } => *bits as f64 + sc(group),
            Fmt::Ternary { group } => 1.6 + sc(group), // 5 trits por byte
            Fmt::Binary { group } => 1.0 + sc(group),
            Fmt::Nf4 { group } => 4.0 + sc(group),
            Fmt::KMeans { bits, group } => *bits as f64 + sc(group),
        }
    }

    /// Cuantiza y reconstruye `w` en su lugar.
    pub fn apply(&self, w: &mut [f32]) {
        match self {
            Fmt::F32 => {}
            Fmt::Float { e, m } => {
                let tab = float_table(*e, *m, ((1i32 << (e - 1)) - 1) as i32);
                w.iter_mut().for_each(|x| *x = round_to(&tab, *x));
            }
            Fmt::MiniFloat { e, m, group } => {
                let tab = float_table(*e, *m, ((1i32 << (e - 1)) - 1) as i32);
                let vmax = *tab.last().unwrap();
                for_blocks(w, *group, |b| {
                    let s = absmax(b) / vmax;
                    if s > 0.0 {
                        b.iter_mut().for_each(|x| *x = round_to(&tab, *x / s) * s);
                    }
                });
            }
            Fmt::Int { bits, group } => {
                let qmax = ((1u32 << (bits - 1)) - 1) as f32;
                for_blocks(w, *group, |b| {
                    let s = absmax(b) / qmax;
                    if s > 0.0 {
                        b.iter_mut().for_each(|x| *x = (*x / s).round().clamp(-qmax, qmax) * s);
                    }
                });
            }
            Fmt::Ternary { group } => for_blocks(w, *group, |b| {
                let delta = 0.7 * b.iter().map(|x| x.abs()).sum::<f32>() / b.len() as f32;
                let (mut sum, mut n) = (0f32, 0usize);
                for x in b.iter() {
                    if x.abs() > delta {
                        sum += x.abs();
                        n += 1;
                    }
                }
                let s = if n > 0 { sum / n as f32 } else { 0.0 };
                b.iter_mut().for_each(|x| *x = if x.abs() > delta { s * x.signum() } else { 0.0 });
            }),
            Fmt::Binary { group } => for_blocks(w, *group, |b| {
                let s = b.iter().map(|x| x.abs()).sum::<f32>() / b.len() as f32;
                b.iter_mut().for_each(|x| *x = if *x >= 0.0 { s } else { -s });
            }),
            Fmt::Nf4 { group } => {
                let tab = NF4.to_vec();
                for_blocks(w, *group, |b| {
                    let s = absmax(b);
                    if s > 0.0 {
                        b.iter_mut().for_each(|x| *x = round_to_signed(&tab, *x / s) * s);
                    }
                });
            }
            Fmt::KMeans { bits, group } => {
                // normalizar por bloque, k-means sobre todos los valores, reconstruir
                let g = if *group == 0 { w.len() } else { *group };
                let scales: Vec<f32> = w.chunks(g).map(absmax).collect();
                let norm: Vec<f32> = w
                    .chunks(g)
                    .zip(&scales)
                    .flat_map(|(b, &s)| b.iter().map(move |x| if s > 0.0 { x / s } else { 0.0 }))
                    .collect();
                let cb = kmeans_1d(&norm, 1usize << bits, 25);
                for (b, &s) in w.chunks_mut(g).zip(&scales) {
                    b.iter_mut().for_each(|x| {
                        let v = if s > 0.0 { *x / s } else { 0.0 };
                        *x = round_to_signed(&cb, v) * s;
                    });
                }
            }
        }
    }
}

fn absmax(b: &[f32]) -> f32 {
    b.iter().fold(0f32, |m, x| m.max(x.abs()))
}

fn for_blocks(w: &mut [f32], group: usize, mut f: impl FnMut(&mut [f32])) {
    let g = if group == 0 { w.len() } else { group };
    w.chunks_mut(g).for_each(|b| f(b));
}

/// Valores no negativos representables por un minifloat (subnormales incluidos, sin inf/nan).
pub fn float_table(e: u32, m: u32, bias: i32) -> Vec<f32> {
    let mut v = Vec::new();
    let emax = if e >= 5 { (1u32 << e) - 1 } else { 1u32 << e }; // IEEE reserva el exponente máximo
    for ex in 0..emax {
        for mt in 0..(1u32 << m) {
            let frac = mt as f64 / (1u64 << m) as f64;
            let val = if ex == 0 {
                frac * 2f64.powi(1 - bias)
            } else {
                (1.0 + frac) * 2f64.powi(ex as i32 - bias)
            };
            v.push(val as f32);
        }
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}

/// Redondeo al valor más cercano de una tabla simétrica de magnitudes (ordenada, ≥ 0).
fn round_to(tab: &[f32], x: f32) -> f32 {
    let a = x.abs();
    let i = tab.partition_point(|&v| v < a);
    let best = if i == 0 {
        tab[0]
    } else if i == tab.len() {
        tab[i - 1]
    } else if a - tab[i - 1] <= tab[i] - a {
        tab[i - 1]
    } else {
        tab[i]
    };
    best * x.signum()
}

/// Redondeo al valor más cercano de una tabla con signo (ordenada).
fn round_to_signed(tab: &[f32], x: f32) -> f32 {
    let i = tab.partition_point(|&v| v < x);
    if i == 0 {
        tab[0]
    } else if i == tab.len() {
        tab[i - 1]
    } else if x - tab[i - 1] <= tab[i] - x {
        tab[i - 1]
    } else {
        tab[i]
    }
}

const NF4: [f32; 16] = [
    -1.0, -0.696_192_8, -0.525_073_05, -0.394_917_5, -0.284_441_38, -0.184_773_43, -0.091_050_036, 0.0,
    0.079_580_3, 0.160_930_2, 0.246_112_3, 0.337_915_24, 0.440_709_83, 0.562_617, 0.722_956_84, 1.0,
];

/// k-means 1D (Lloyd) con inicialización por cuantiles. Devuelve centroides ordenados.
pub fn kmeans_1d(xs: &[f32], k: usize, iters: usize) -> Vec<f32> {
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let mut c: Vec<f32> = (0..k).map(|i| sorted[((i as f64 + 0.5) / k as f64 * n as f64) as usize]).collect();
    for _ in 0..iters {
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        c.dedup();
        let mut sum = vec![0f64; c.len()];
        let mut cnt = vec![0usize; c.len()];
        // asignación sobre datos ordenados: fronteras = puntos medios
        let mut j = 0;
        for &x in &sorted {
            while j + 1 < c.len() && x > 0.5 * (c[j] + c[j + 1]) {
                j += 1;
            }
            sum[j] += x as f64;
            cnt[j] += 1;
        }
        for i in 0..c.len() {
            if cnt[i] > 0 {
                c[i] = (sum[i] / cnt[i] as f64) as f32;
            }
        }
    }
    c.sort_by(|a, b| a.partial_cmp(b).unwrap());
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_table_matches_ieee() {
        let t = float_table(5, 10, 15);
        assert_eq!(*t.last().unwrap(), 65504.0);
        assert_eq!(t[1], 2f32.powi(-24));
    }

    #[test]
    fn e2m1_is_fp4() {
        let t = float_table(2, 1, 1);
        assert_eq!(t, vec![0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn int8_error_small() {
        let mut w: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) / 40.0).collect();
        let orig = w.clone();
        Fmt::Int { bits: 8, group: 64 }.apply(&mut w);
        for (a, b) in w.iter().zip(&orig) {
            assert!((a - b).abs() < 0.01);
        }
    }
}
