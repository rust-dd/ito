use ndarray::Array1;
use stochastic_rs_stochastic::ProcessExt;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::simd_rng::Unseeded;

fn main() {
    let gbm = Gbm::<f64>::new(0.05, 0.2, 20, Some(100.0), Some(1.0), Unseeded);
    let path: Array1<f64> = gbm.sample();
    println!("len={} first={:.4} last={:.4}", path.len(), path[0], path[path.len() - 1]);
    let many: Vec<Array1<f64>> = gbm.sample_par(3);
    println!("paths={}", many.len());
}
