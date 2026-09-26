use lengua_ligera::model::{Acts, Config, Model};
use lengua_ligera::rng::Rng;

#[test]
fn analytic_gradient_matches_finite_differences() {
    let cfg = Config { vocab: 16, ctx: 8, d: 16, layers: 2, heads: 2 };
    let mut m = Model::init(cfg, 7);
    // pesos más grandes para que los gradientes no sean diminutos
    let mut rng = Rng::new(3);
    for x in m.p.iter_mut() {
        *x += rng.normal() * 0.1;
    }
    let inp: Vec<u8> = (0..8).map(|i| (i * 5 % 16) as u8).collect();
    let tgt: Vec<u8> = (0..8).map(|i| (i * 3 % 16 + 1) as u8 % 16).collect();
    let mut a = Acts::new(&cfg);
    let wt = m.transposed();
    m.forward(&wt, &mut a, &inp, Some(&tgt));
    let mut g = vec![0f32; m.p.len()];
    m.backward(&wt, &mut a, &inp, &tgt, &mut g, 1.0 / 8.0);

    let loss_at = |m: &Model| {
        let mut a = Acts::new(&cfg);
        m.forward(&m.transposed(), &mut a, &inp, Some(&tgt)) as f64
    };
    let mut worst = 0f64;
    let mut checked = 0;
    for idx in (0..m.p.len()).step_by(7) {
        let old = m.p[idx];
        let eps = 1e-2f32;
        m.p[idx] = old + eps;
        let lp = loss_at(&m);
        m.p[idx] = old - eps;
        let lm = loss_at(&m);
        m.p[idx] = old;
        let num = (lp - lm) / (2.0 * eps as f64);
        let ana = g[idx] as f64;
        if num.abs().max(ana.abs()) < 1e-3 {
            continue;
        }
        let rel = (num - ana).abs() / num.abs().max(ana.abs());
        worst = worst.max(rel);
        checked += 1;
        assert!(rel < 0.05, "param {idx}: numérico {num:.6} vs analítico {ana:.6}");
    }
    assert!(checked > 100, "sólo {checked} parámetros revisados");
    eprintln!("gradcheck: {checked} parámetros, peor error relativo {worst:.4}");
}
