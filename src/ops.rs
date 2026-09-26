//! Núcleos numéricos: matmul con bloques en registros, transposición, GELU.

/// y[t×n] += x[t×k] · w[k×n], todo row-major. Requiere n % 16 == 0.
pub fn mm_acc(y: &mut [f32], x: &[f32], w: &[f32], t: usize, k: usize, n: usize) {
    assert!(n % 16 == 0, "mm_acc: n={n} no es múltiplo de 16");
    assert!(y.len() >= t * n && x.len() >= t * k && w.len() >= k * n);
    const KB: usize = 128;
    let mut k0 = 0;
    while k0 < k {
        let k1 = (k0 + KB).min(k);
        let mut j = 0;
        while j < n {
            let mut i = 0;
            while i + 4 <= t {
                let mut acc = [[0f32; 16]; 4];
                for kk in k0..k1 {
                    let wr: &[f32; 16] = w[kk * n + j..kk * n + j + 16].try_into().unwrap();
                    let a = [x[i * k + kk], x[(i + 1) * k + kk], x[(i + 2) * k + kk], x[(i + 3) * k + kk]];
                    for r in 0..4 {
                        for c in 0..16 {
                            acc[r][c] += a[r] * wr[c];
                        }
                    }
                }
                for r in 0..4 {
                    let yr = &mut y[(i + r) * n + j..(i + r) * n + j + 16];
                    for c in 0..16 {
                        yr[c] += acc[r][c];
                    }
                }
                i += 4;
            }
            while i < t {
                let mut acc = [0f32; 16];
                for kk in k0..k1 {
                    let wr: &[f32; 16] = w[kk * n + j..kk * n + j + 16].try_into().unwrap();
                    let a = x[i * k + kk];
                    for c in 0..16 {
                        acc[c] += a * wr[c];
                    }
                }
                let yr = &mut y[i * n + j..i * n + j + 16];
                for c in 0..16 {
                    yr[c] += acc[c];
                }
                i += 1;
            }
            j += 16;
        }
        k0 = k1;
    }
}

/// Transpone src (rows×cols) a (cols×rows).
pub fn transpose(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    const B: usize = 32;
    for r0 in (0..rows).step_by(B) {
        for c0 in (0..cols).step_by(B) {
            for r in r0..(r0 + B).min(rows) {
                for c in c0..(c0 + B).min(cols) {
                    out[c * rows + r] = src[r * cols + c];
                }
            }
        }
    }
    out
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

const GELU_C: f32 = 0.797_884_6; // sqrt(2/pi)

#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (GELU_C * (x + 0.044715 * x * x * x)).tanh())
}

#[inline]
pub fn gelu_grad(x: f32) -> f32 {
    let u = GELU_C * (x + 0.044715 * x * x * x);
    let th = u.tanh();
    0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * GELU_C * (1.0 + 3.0 * 0.044715 * x * x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mm_matches_naive() {
        let (t, k, n) = (7, 37, 48);
        let x: Vec<f32> = (0..t * k).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
        let w: Vec<f32> = (0..k * n).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1).collect();
        let mut y = vec![1.0f32; t * n];
        mm_acc(&mut y, &x, &w, t, k, n);
        for i in 0..t {
            for j in 0..n {
                let mut s = 1.0f32;
                for kk in 0..k {
                    s += x[i * k + kk] * w[kk * n + j];
                }
                assert!((s - y[i * n + j]).abs() < 1e-4);
            }
        }
    }
}
