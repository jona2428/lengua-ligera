use lengua_ligera::ops::mm_acc;
use std::time::Instant;
fn main() {
    for &(t, k, n) in &[(128usize, 256usize, 768usize), (128, 256, 1024), (128, 1024, 256), (256, 128, 256), (1024, 128, 256)] {
        let x = vec![0.5f32; t * k];
        let w = vec![0.25f32; k * n];
        let mut y = vec![0f32; t * n];
        let reps = 200;
        let s = Instant::now();
        for _ in 0..reps { mm_acc(&mut y, &x, &w, t, k, n); }
        let el = s.elapsed().as_secs_f64();
        println!("t{t} k{k} n{n}: {:.1} GFLOP/s", 2.0 * (t * k * n * reps) as f64 / el / 1e9);
    }
}

#[allow(dead_code)]
pub fn unused() {}
