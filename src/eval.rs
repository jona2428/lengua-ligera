//! Evaluación: pérdida media sobre ventanas disjuntas del set de validación.
use crate::model::{Acts, Model};

/// Pérdida media (nats/carácter) sobre las primeras `chars` posiciones de `valid`.
pub fn evaluate(m: &Model, valid: &[u8], chars: usize, threads: usize) -> f64 {
    let t = m.cfg.ctx;
    let n_win = chars.min(valid.len() - 1) / t;
    let wt = m.transposed();
    let sums: Vec<f64> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|w| {
                let wt = &wt;
                s.spawn(move || {
                    let mut a = Acts::new(&m.cfg);
                    let mut sum = 0f64;
                    for k in (w..n_win).step_by(threads) {
                        let o = k * t;
                        sum += m.forward(wt, &mut a, &valid[o..o + t], Some(&valid[o + 1..o + t + 1])) as f64;
                    }
                    sum
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    sums.iter().sum::<f64>() / n_win as f64
}

pub fn bpc(loss_nats: f64) -> f64 {
    loss_nats / std::f64::consts::LN_2
}
