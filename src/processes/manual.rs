//! Hand-written registrations for processes whose constructors take an argument
//! the scalar/vector form can't express — here a fixed enum variant. The scalar
//! parameters are still exposed; the non-scalar argument is pinned to its
//! conventional value.

use ndarray::Array1;
use ndarray::Array2;
use rand::Rng;
use rand_distr::Distribution;
use rand_distr::Normal;
use stochastic_rs_stochastic::correlation::transformed_ou::TransformedOU;
use stochastic_rs_stochastic::correlation::transformed_ou::Transformation;
use stochastic_rs_stochastic::diffusion::cfou::Cfou;
use stochastic_rs_stochastic::jump::bates::Bates1996;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::jump::kou::Kou;
use stochastic_rs_stochastic::jump::levy_diffusion::LevyDiffusion;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::process::cpoisson::CompoundPoisson;
use stochastic_rs_stochastic::process::multivariate_hawkes::MultivariateHawkes;
use stochastic_rs_stochastic::process::poisson::Poisson;
use stochastic_rs_stochastic::simd_rng::Unseeded;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::heston::Heston;

use crate::registry::Category;
use crate::registry::ChartSource;
use crate::registry::ParamDefault;
use crate::registry::ParamKind;
use crate::registry::ParamSpec;
use crate::registry::ParamValues;
use crate::registry::ProcessDescriptor;
use crate::registry::adapters::ComplexPath;
use crate::registry::adapters::MultiDim;
use crate::registry::adapters::Path1D;
use crate::registry::adapters::VecPath;

fn build_heston(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(MultiDim {
        process: Heston::<f64>::new(
            values.opt_f64("s0"),
            values.opt_f64("v0"),
            values.f64("kappa"),
            values.f64("theta"),
            values.f64("sigma"),
            values.f64("rho"),
            values.f64("mu"),
            values.usize("n"),
            values.opt_f64("t"),
            HestonPow::Sqrt,
            values.opt_bool("use_sym"),
            Unseeded,
        ),
        components: &["asset", "variance"],
    })
}

inventory::submit! {
    ProcessDescriptor {
        name: "Heston",
        category: Category::Volatility,
        params: &[
            ParamSpec { name: "s0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(100.0)), doc: "Initial spot" },
            ParamSpec { name: "v0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.04)), doc: "Initial variance" },
            ParamSpec { name: "kappa", kind: ParamKind::F64, default: ParamDefault::F64(1.5), doc: "Reversion speed" },
            ParamSpec { name: "theta", kind: ParamKind::F64, default: ParamDefault::F64(0.04), doc: "Long-run variance" },
            ParamSpec { name: "sigma", kind: ParamKind::F64, default: ParamDefault::F64(0.3), doc: "Vol of vol" },
            ParamSpec { name: "rho", kind: ParamKind::F64, default: ParamDefault::F64(-0.7), doc: "Correlation" },
            ParamSpec { name: "mu", kind: ParamKind::F64, default: ParamDefault::F64(0.05), doc: "Drift" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
            ParamSpec { name: "use_sym", kind: ParamKind::OptBool, default: ParamDefault::OptBool(Some(true)), doc: "Symmetrise" },
        ],
        build: build_heston,
    }
}

fn build_transformed_ou(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(Path1D(TransformedOU::<f64>::new(
        values.f64("kappa"),
        values.f64("mu"),
        values.f64("sigma"),
        values.f64("rho0"),
        Transformation::Tanh,
        values.usize("n"),
        values.opt_f64("t"),
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "TransformedOU",
        category: Category::Correlation,
        params: &[
            ParamSpec { name: "kappa", kind: ParamKind::F64, default: ParamDefault::F64(1.5), doc: "Reversion speed" },
            ParamSpec { name: "mu", kind: ParamKind::F64, default: ParamDefault::F64(0.0), doc: "Long-run mean" },
            ParamSpec { name: "sigma", kind: ParamKind::F64, default: ParamDefault::F64(0.3), doc: "Diffusion scale" },
            ParamSpec { name: "rho0", kind: ParamKind::F64, default: ParamDefault::F64(0.0), doc: "Initial correlation" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_transformed_ou,
    }
}

fn build_merton(values: &ParamValues) -> Box<dyn ChartSource> {
    let jumps = Normal::<f64>::new(values.f64("jump_mean"), values.f64("jump_std").max(1e-9)).unwrap();
    let cpoisson = CompoundPoisson::new(
        jumps,
        Poisson::new(
            values.f64("lambda"),
            Some(values.usize("n")),
            values.opt_f64("t"),
            Unseeded,
        ),
        Unseeded,
    );
    Box::new(Path1D(Merton::new(
        values.f64("alpha"),
        values.f64("sigma"),
        values.f64("lambda"),
        values.f64("theta"),
        values.usize("n"),
        values.opt_f64("x0"),
        values.opt_f64("t"),
        cpoisson,
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "Merton",
        category: Category::Jump,
        params: &[
            ParamSpec { name: "alpha", kind: ParamKind::F64, default: ParamDefault::F64(0.05), doc: "Drift" },
            ParamSpec { name: "sigma", kind: ParamKind::F64, default: ParamDefault::F64(0.2), doc: "Diffusion scale" },
            ParamSpec { name: "lambda", kind: ParamKind::F64, default: ParamDefault::F64(1.0), doc: "Jump intensity" },
            ParamSpec { name: "theta", kind: ParamKind::F64, default: ParamDefault::F64(0.0), doc: "Jump drift" },
            ParamSpec { name: "jump_mean", kind: ParamKind::F64, default: ParamDefault::F64(-0.05), doc: "Mean jump size (Normal)" },
            ParamSpec { name: "jump_std", kind: ParamKind::F64, default: ParamDefault::F64(0.12), doc: "Jump size std (Normal)" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "x0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(100.0)), doc: "Initial value" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_merton,
    }
}

/// Compound-Poisson source with i.i.d. Normal jump sizes (Merton/Bates/Lévy).
fn normal_cpoisson(
    mean: f64,
    std: f64,
    lambda: f64,
    n: usize,
    t: Option<f64>,
) -> CompoundPoisson<f64, Normal<f64>, Unseeded> {
    CompoundPoisson::new(
        Normal::new(mean, std.max(1e-9)).unwrap(),
        Poisson::new(lambda, Some(n), t, Unseeded),
        Unseeded,
    )
}

/// Kou's asymmetric double-exponential jump size: with probability `p` an
/// upward `Exp(eta_up)` jump, otherwise a downward `-Exp(eta_down)` jump.
struct DoubleExp {
    p: f64,
    eta_up: f64,
    eta_down: f64,
}

impl Distribution<f64> for DoubleExp {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let coin: f64 = rng.random();
        let exp1 = -(1.0 - rng.random::<f64>()).ln();
        if coin < self.p {
            exp1 / self.eta_up
        } else {
            -exp1 / self.eta_down
        }
    }
}

fn build_kou(values: &ParamValues) -> Box<dyn ChartSource> {
    let cpoisson = CompoundPoisson::new(
        DoubleExp {
            p: values.f64("p"),
            eta_up: values.f64("eta_up").max(1.01),
            eta_down: values.f64("eta_down").max(0.01),
        },
        Poisson::new(values.f64("lambda"), Some(values.usize("n")), values.opt_f64("t"), Unseeded),
        Unseeded,
    );
    Box::new(Path1D(Kou::new(
        values.f64("alpha"),
        values.f64("sigma"),
        values.f64("lambda"),
        values.f64("theta"),
        values.usize("n"),
        values.opt_f64("x0"),
        values.opt_f64("t"),
        cpoisson,
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "Kou",
        category: Category::Jump,
        params: &[
            ParamSpec { name: "alpha", kind: ParamKind::F64, default: ParamDefault::F64(0.05), doc: "Drift" },
            ParamSpec { name: "sigma", kind: ParamKind::F64, default: ParamDefault::F64(0.2), doc: "Diffusion scale" },
            ParamSpec { name: "lambda", kind: ParamKind::F64, default: ParamDefault::F64(1.0), doc: "Jump intensity" },
            ParamSpec { name: "theta", kind: ParamKind::F64, default: ParamDefault::F64(0.0), doc: "Jump drift" },
            ParamSpec { name: "p", kind: ParamKind::F64, default: ParamDefault::F64(0.4), doc: "Upward-jump probability" },
            ParamSpec { name: "eta_up", kind: ParamKind::F64, default: ParamDefault::F64(10.0), doc: "Upward decay (>1)" },
            ParamSpec { name: "eta_down", kind: ParamKind::F64, default: ParamDefault::F64(5.0), doc: "Downward decay" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "x0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(100.0)), doc: "Initial value" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_kou,
    }
}

fn build_levy_diffusion(values: &ParamValues) -> Box<dyn ChartSource> {
    let cpoisson = normal_cpoisson(
        values.f64("jump_mean"),
        values.f64("jump_std"),
        values.f64("lambda"),
        values.usize("n"),
        values.opt_f64("t"),
    );
    Box::new(Path1D(LevyDiffusion::new(
        values.f64("gamma"),
        values.f64("sigma"),
        values.usize("n"),
        values.opt_f64("x0"),
        values.opt_f64("t"),
        cpoisson,
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "LevyDiffusion",
        category: Category::Jump,
        params: &[
            ParamSpec { name: "gamma", kind: ParamKind::F64, default: ParamDefault::F64(0.05), doc: "Drift" },
            ParamSpec { name: "sigma", kind: ParamKind::F64, default: ParamDefault::F64(0.2), doc: "Diffusion scale" },
            ParamSpec { name: "lambda", kind: ParamKind::F64, default: ParamDefault::F64(1.0), doc: "Jump intensity" },
            ParamSpec { name: "jump_mean", kind: ParamKind::F64, default: ParamDefault::F64(0.0), doc: "Mean jump size" },
            ParamSpec { name: "jump_std", kind: ParamKind::F64, default: ParamDefault::F64(0.15), doc: "Jump size std" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "x0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.0)), doc: "Initial value" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_levy_diffusion,
    }
}

fn build_jump_fou(values: &ParamValues) -> Box<dyn ChartSource> {
    let cpoisson = normal_cpoisson(
        values.f64("jump_mean"),
        values.f64("jump_std"),
        values.f64("lambda"),
        values.usize("n"),
        values.opt_f64("t"),
    );
    Box::new(Path1D(JumpFou::new(
        values.f64("hurst"),
        values.f64("theta"),
        values.f64("mu"),
        values.f64("sigma"),
        values.usize("n"),
        values.opt_f64("x0"),
        values.opt_f64("t"),
        cpoisson,
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "JumpFou",
        category: Category::Jump,
        params: &[
            ParamSpec { name: "hurst", kind: ParamKind::F64, default: ParamDefault::F64(0.7), doc: "Hurst exponent" },
            ParamSpec { name: "theta", kind: ParamKind::F64, default: ParamDefault::F64(1.0), doc: "Reversion speed" },
            ParamSpec { name: "mu", kind: ParamKind::F64, default: ParamDefault::F64(1.0), doc: "Long-run mean" },
            ParamSpec { name: "sigma", kind: ParamKind::F64, default: ParamDefault::F64(0.3), doc: "Diffusion scale" },
            ParamSpec { name: "lambda", kind: ParamKind::F64, default: ParamDefault::F64(1.0), doc: "Jump intensity" },
            ParamSpec { name: "jump_mean", kind: ParamKind::F64, default: ParamDefault::F64(0.0), doc: "Mean jump size" },
            ParamSpec { name: "jump_std", kind: ParamKind::F64, default: ParamDefault::F64(0.15), doc: "Jump size std" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "x0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.5)), doc: "Initial value" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_jump_fou,
    }
}

fn build_bates(values: &ParamValues) -> Box<dyn ChartSource> {
    let cpoisson = normal_cpoisson(
        values.f64("jump_mean"),
        values.f64("jump_std"),
        values.f64("lambda"),
        values.usize("n"),
        values.opt_f64("t"),
    );
    Box::new(MultiDim {
        process: Bates1996::new(
            values.opt_f64("mu"),
            values.opt_f64("b"),
            values.opt_f64("r"),
            values.opt_f64("r_f"),
            values.f64("lambda"),
            values.f64("k"),
            values.f64("alpha"),
            values.f64("beta"),
            values.f64("sigma"),
            values.f64("rho"),
            values.usize("n"),
            values.opt_f64("s0"),
            values.opt_f64("v0"),
            values.opt_f64("t"),
            values.opt_bool("use_sym"),
            cpoisson,
            Unseeded,
        ),
        components: &["asset", "variance"],
    })
}

inventory::submit! {
    ProcessDescriptor {
        name: "Bates1996",
        category: Category::Volatility,
        params: &[
            ParamSpec { name: "mu", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.05)), doc: "Drift" },
            ParamSpec { name: "b", kind: ParamKind::OptF64, default: ParamDefault::OptF64(None), doc: "Cost of carry" },
            ParamSpec { name: "r", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.03)), doc: "Rate" },
            ParamSpec { name: "r_f", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.0)), doc: "Foreign rate" },
            ParamSpec { name: "lambda", kind: ParamKind::F64, default: ParamDefault::F64(1.0), doc: "Jump intensity" },
            ParamSpec { name: "k", kind: ParamKind::F64, default: ParamDefault::F64(0.0), doc: "Mean jump" },
            ParamSpec { name: "alpha", kind: ParamKind::F64, default: ParamDefault::F64(1.5), doc: "Reversion speed" },
            ParamSpec { name: "beta", kind: ParamKind::F64, default: ParamDefault::F64(0.04), doc: "Long-run variance" },
            ParamSpec { name: "sigma", kind: ParamKind::F64, default: ParamDefault::F64(0.3), doc: "Vol of vol" },
            ParamSpec { name: "rho", kind: ParamKind::F64, default: ParamDefault::F64(-0.7), doc: "Correlation" },
            ParamSpec { name: "jump_mean", kind: ParamKind::F64, default: ParamDefault::F64(-0.05), doc: "Mean jump size" },
            ParamSpec { name: "jump_std", kind: ParamKind::F64, default: ParamDefault::F64(0.1), doc: "Jump size std" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "s0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(100.0)), doc: "Initial spot" },
            ParamSpec { name: "v0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.04)), doc: "Initial variance" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
            ParamSpec { name: "use_sym", kind: ParamKind::OptBool, default: ParamDefault::OptBool(Some(true)), doc: "Symmetrise" },
        ],
        build: build_bates,
    }
}

fn build_cfou(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(ComplexPath(Cfou::<f64>::new(
        values.f64("hurst"),
        values.f64("lambda"),
        values.f64("omega"),
        values.f64("a"),
        values.usize("n"),
        values.opt_f64("x1_0"),
        values.opt_f64("x2_0"),
        values.opt_f64("t"),
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "Cfou",
        category: Category::Diffusion,
        params: &[
            ParamSpec { name: "hurst", kind: ParamKind::F64, default: ParamDefault::F64(0.7), doc: "Hurst exponent" },
            ParamSpec { name: "lambda", kind: ParamKind::F64, default: ParamDefault::F64(1.0), doc: "Reversion speed" },
            ParamSpec { name: "omega", kind: ParamKind::F64, default: ParamDefault::F64(1.0), doc: "Angular frequency" },
            ParamSpec { name: "a", kind: ParamKind::F64, default: ParamDefault::F64(1.0), doc: "Amplitude" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "x1_0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.5)), doc: "Initial real part" },
            ParamSpec { name: "x2_0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.5)), doc: "Initial imag part" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_cfou,
    }
}

fn square_matrix(values: &ParamValues, name: &str, dim: usize) -> Array2<f64> {
    let flat = values.f64vec(name);
    Array2::from_shape_vec((dim, dim), flat).unwrap_or_else(|_| Array2::zeros((dim, dim)))
}

fn build_multivariate_hawkes(values: &ParamValues) -> Box<dyn ChartSource> {
    let mu = Array1::from_vec(values.f64vec("mu"));
    let dim = mu.len().max(1);
    Box::new(VecPath(MultivariateHawkes::<f64>::new(
        mu,
        square_matrix(values, "alpha", dim),
        square_matrix(values, "beta", dim),
        values.f64("t_max"),
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "MultivariateHawkes",
        category: Category::Process,
        params: &[
            ParamSpec { name: "mu", kind: ParamKind::F64Vec, default: ParamDefault::F64Vec(&[0.5, 0.5]), doc: "Baseline intensities (one per dimension)" },
            ParamSpec { name: "alpha", kind: ParamKind::F64Vec, default: ParamDefault::F64Vec(&[0.2, 0.1, 0.1, 0.2]), doc: "Excitation matrix, row-major" },
            ParamSpec { name: "beta", kind: ParamKind::F64Vec, default: ParamDefault::F64Vec(&[1.0, 1.0, 1.0, 1.0]), doc: "Decay matrix, row-major" },
            ParamSpec { name: "t_max", kind: ParamKind::F64, default: ParamDefault::F64(10.0), doc: "Time horizon" },
        ],
        build: build_multivariate_hawkes,
    }
}
