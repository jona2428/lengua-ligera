use lengua_ligera::model::{Acts, Config, Model};
use std::time::Instant;
fn main() {
    let cfg = Config { vocab: 128, ctx: 128, d: 256, layers: 4, heads: 4 };
    let m = Model::init(cfg, 1);
    let mut a = Acts::new(&cfg);
    let inp: Vec<u8> = (0..128).map(|i| (i * 7 % 128) as u8).collect();
    let tgt: Vec<u8> = (0..128).map(|i| (i * 3 % 128) as u8).collect();
    let mut g = vec![0f32; m.p.len()];
    let s = Instant::now(); let wt = m.transposed(); println!("transposed {:.2} ms", s.elapsed().as_secs_f64()*1e3);
    let reps = 20;
    let s = Instant::now(); for _ in 0..reps { m.forward(&wt, &mut a, &inp, Some(&tgt)); } let f = s.elapsed().as_secs_f64()/reps as f64;
    let s = Instant::now(); for _ in 0..reps { m.forward(&wt, &mut a, &inp, Some(&tgt)); m.backward(&wt, &mut a, &inp, &tgt, &mut g, 1e-3); } let fb = s.elapsed().as_secs_f64()/reps as f64;
    println!("forward {:.1} ms  backward {:.1} ms  (ideal fwd a 50 GFLOP/s: {:.1} ms)", f*1e3, (fb-f)*1e3, 2.0*3.2e6*128.0/50e9*1e3);
}
