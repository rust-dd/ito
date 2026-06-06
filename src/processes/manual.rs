//! Hand-written registrations for processes whose constructors take an argument
//! the scalar/vector form can't express — here a fixed enum variant. The scalar
//! parameters are still exposed; the non-scalar argument is pinned to its
//! conventional value.

use ndarray::Array1;
use ndarray::Array2;
use rand::Rng;
use rand_distr::Distribution;
use rand_distr::Exp;
use rand_distr::Normal;
use stochastic_rs_stochastic::correlation::transformed_ou::Transformation;
use stochastic_rs_stochastic::correlation::transformed_ou::TransformedOU;
use stochastic_rs_stochastic::diffusion::cfou::Cfou;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::regime_switching::RegimeSwitchingDiffusion;
use stochastic_rs_stochastic::interest::adg::Adg;
use stochastic_rs_stochastic::interest::cir_2f::Cir2F;
use stochastic_rs_stochastic::interest::hjm::Hjm;
use stochastic_rs_stochastic::interest::ho_lee::HoLee;
use stochastic_rs_stochastic::interest::hull_white::HullWhite;
use stochastic_rs_stochastic::interest::hull_white_2f::HullWhite2F;
use stochastic_rs_stochastic::jump::bates::Bates1996;
use stochastic_rs_stochastic::jump::bilateral_gamma::BilateralGammaMotion;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::jump::jump_fou_custom::JumpFOUCustom;
use stochastic_rs_stochastic::jump::kou::Kou;
use stochastic_rs_stochastic::jump::levy_diffusion::LevyDiffusion;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::process::ccustom::CompoundCustom;
use stochastic_rs_stochastic::process::cpoisson::CompoundPoisson;
use stochastic_rs_stochastic::process::customjt::CustomJt;
use stochastic_rs_stochastic::process::multivariate_hawkes::MultivariateHawkes;
use stochastic_rs_stochastic::process::poisson::Poisson;
use stochastic_rs_stochastic::process::subordinator::ctrw::Ctrw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwJumpLaw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwWaitingLaw;
use stochastic_rs_stochastic::process::volterra::Volterra;
use stochastic_rs_stochastic::process::volterra::VolterraKernel;
use stochastic_rs_stochastic::simd_rng::Unseeded;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::heston::Heston;
use stochastic_rs_stochastic::volatility::heston2d::Heston2D;
use stochastic_rs_stochastic::volatility::multifactor_heston::MultifactorHeston;
use stochastic_rs_stochastic::volatility::multifactor_sabr::MultifactorSabr;

use crate::registry::Category;
use crate::registry::ChartSource;
use crate::registry::ParamDefault;
use crate::registry::ParamKind;
use crate::registry::ParamSpec;
use crate::registry::ParamValues;
use crate::registry::ProcessDescriptor;
use crate::registry::adapters::ComplexPath;
use crate::registry::adapters::Curve;
use crate::registry::adapters::MultiDim;
use crate::registry::adapters::Path1D;
use crate::registry::adapters::TupleDim;
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
    let jumps =
        Normal::<f64>::new(values.f64("jump_mean"), values.f64("jump_std").max(1e-9)).unwrap();
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
        Poisson::new(
            values.f64("lambda"),
            Some(values.usize("n")),
            values.opt_f64("t"),
            Unseeded,
        ),
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

/// Flat 3% forward curve used to pin the callable term-structure arguments of
/// the Hull-White / Ho-Lee / 2-factor models.
fn flat_curve(_: f64) -> f64 {
    0.03
}

fn build_hull_white(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(Path1D(HullWhite::<f64>::new(
        flat_curve as fn(f64) -> f64,
        values.f64("alpha"),
        values.f64("sigma"),
        values.usize("n"),
        values.opt_f64("x0"),
        values.opt_f64("t"),
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "HullWhite",
        category: Category::Interest,
        params: &[
            ParamSpec { name: "alpha", kind: ParamKind::F64, default: ParamDefault::F64(0.1), doc: "Reversion speed" },
            ParamSpec { name: "sigma", kind: ParamKind::F64, default: ParamDefault::F64(0.01), doc: "Diffusion scale" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "x0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.03)), doc: "Initial rate" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_hull_white,
    }
}

fn build_ho_lee(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(Path1D(HoLee::<f64>::new(
        None,
        values.opt_f64("theta"),
        values.f64("sigma"),
        values.usize("n"),
        values.opt_f64("t"),
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "HoLee",
        category: Category::Interest,
        params: &[
            ParamSpec { name: "theta", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.02)), doc: "Drift" },
            ParamSpec { name: "sigma", kind: ParamKind::F64, default: ParamDefault::F64(0.01), doc: "Diffusion scale" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_ho_lee,
    }
}

fn build_hull_white_2f(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(MultiDim {
        process: HullWhite2F::<f64>::new(
            flat_curve as fn(f64) -> f64,
            values.f64("theta"),
            values.f64("sigma1"),
            values.f64("sigma2"),
            values.f64("rho"),
            values.f64("b"),
            values.opt_f64("x0"),
            values.opt_f64("t"),
            values.usize("n"),
            Unseeded,
        ),
        components: &["factor 1", "factor 2"],
    })
}

inventory::submit! {
    ProcessDescriptor {
        name: "HullWhite2F",
        category: Category::Interest,
        params: &[
            ParamSpec { name: "theta", kind: ParamKind::F64, default: ParamDefault::F64(0.03), doc: "Long-run level" },
            ParamSpec { name: "sigma1", kind: ParamKind::F64, default: ParamDefault::F64(0.01), doc: "Factor 1 vol" },
            ParamSpec { name: "sigma2", kind: ParamKind::F64, default: ParamDefault::F64(0.005), doc: "Factor 2 vol" },
            ParamSpec { name: "rho", kind: ParamKind::F64, default: ParamDefault::F64(-0.5), doc: "Correlation" },
            ParamSpec { name: "b", kind: ParamKind::F64, default: ParamDefault::F64(0.1), doc: "Second reversion" },
            ParamSpec { name: "x0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.03)), doc: "Initial rate" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
        ],
        build: build_hull_white_2f,
    }
}

fn build_cir_2f(values: &ParamValues) -> Box<dyn ChartSource> {
    let x = Cir::<f64>::new(
        values.f64("x_theta"),
        values.f64("x_mu"),
        values.f64("x_sigma"),
        values.usize("n"),
        values.opt_f64("x_x0"),
        values.opt_f64("t"),
        Some(true),
        Unseeded,
    );
    let y = Cir::<f64>::new(
        values.f64("y_theta"),
        values.f64("y_mu"),
        values.f64("y_sigma"),
        values.usize("n"),
        values.opt_f64("y_x0"),
        values.opt_f64("t"),
        Some(true),
        Unseeded,
    );
    Box::new(Path1D(Cir2F::new(
        x,
        y,
        flat_curve as fn(f64) -> f64,
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "Cir2F",
        category: Category::Interest,
        params: &[
            ParamSpec { name: "x_theta", kind: ParamKind::F64, default: ParamDefault::F64(2.0), doc: "Factor 1 reversion" },
            ParamSpec { name: "x_mu", kind: ParamKind::F64, default: ParamDefault::F64(0.03), doc: "Factor 1 mean" },
            ParamSpec { name: "x_sigma", kind: ParamKind::F64, default: ParamDefault::F64(0.05), doc: "Factor 1 vol" },
            ParamSpec { name: "x_x0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.03)), doc: "Factor 1 start" },
            ParamSpec { name: "y_theta", kind: ParamKind::F64, default: ParamDefault::F64(1.0), doc: "Factor 2 reversion" },
            ParamSpec { name: "y_mu", kind: ParamKind::F64, default: ParamDefault::F64(0.02), doc: "Factor 2 mean" },
            ParamSpec { name: "y_sigma", kind: ParamKind::F64, default: ParamDefault::F64(0.03), doc: "Factor 2 vol" },
            ParamSpec { name: "y_x0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.02)), doc: "Factor 2 start" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_cir_2f,
    }
}

fn build_jump_fou_custom(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(Path1D(JumpFOUCustom::new(
        values.f64("hurst"),
        values.f64("theta"),
        values.f64("mu"),
        values.f64("sigma"),
        values.usize("n"),
        values.opt_f64("x0"),
        values.opt_f64("t"),
        Exp::new(values.f64("lambda").max(0.01)).unwrap(),
        Exp::new((1.0 / values.f64("jump_scale").max(0.01)).max(0.01)).unwrap(),
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "JumpFOUCustom",
        category: Category::Jump,
        params: &[
            ParamSpec { name: "hurst", kind: ParamKind::F64, default: ParamDefault::F64(0.7), doc: "Hurst exponent" },
            ParamSpec { name: "theta", kind: ParamKind::F64, default: ParamDefault::F64(1.0), doc: "Reversion speed" },
            ParamSpec { name: "mu", kind: ParamKind::F64, default: ParamDefault::F64(1.0), doc: "Long-run mean" },
            ParamSpec { name: "sigma", kind: ParamKind::F64, default: ParamDefault::F64(0.3), doc: "Diffusion scale" },
            ParamSpec { name: "lambda", kind: ParamKind::F64, default: ParamDefault::F64(20.0), doc: "Jump-time rate" },
            ParamSpec { name: "jump_scale", kind: ParamKind::F64, default: ParamDefault::F64(0.1), doc: "Mean jump size" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "x0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.5)), doc: "Initial value" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_jump_fou_custom,
    }
}

/// Flat 2-D surface used to pin the callable `Fn2D` arguments of the HJM/ADG
/// term-structure models to a small constant.
fn flat_surface(_: f64, _: f64) -> f64 {
    0.01
}

fn build_adg(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(Curve(Adg::<f64>::new(
        flat_curve as fn(f64) -> f64,
        flat_curve as fn(f64) -> f64,
        Array1::from_vec(values.f64vec("sigma")),
        flat_curve as fn(f64) -> f64,
        flat_curve as fn(f64) -> f64,
        flat_curve as fn(f64) -> f64,
        values.usize("n"),
        values.usize("xn"),
        Array1::from_vec(values.f64vec("x0")),
        values.opt_f64("t"),
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "Adg",
        category: Category::Interest,
        params: &[
            ParamSpec { name: "sigma", kind: ParamKind::F64Vec, default: ParamDefault::F64Vec(&[0.01, 0.01]), doc: "Per-factor vols" },
            ParamSpec { name: "x0", kind: ParamKind::F64Vec, default: ParamDefault::F64Vec(&[0.03, 0.03]), doc: "Initial rates" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "xn", kind: ParamKind::Usize, default: ParamDefault::Usize(2), doc: "Number of factors" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_adg,
    }
}

fn build_hjm(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(MultiDim {
        process: Hjm::<f64>::new(
            flat_curve as fn(f64) -> f64,
            flat_curve as fn(f64) -> f64,
            flat_surface as fn(f64, f64) -> f64,
            flat_surface as fn(f64, f64) -> f64,
            flat_surface as fn(f64, f64) -> f64,
            flat_surface as fn(f64, f64) -> f64,
            flat_surface as fn(f64, f64) -> f64,
            values.usize("n"),
            values.opt_f64("r0"),
            values.opt_f64("p0"),
            values.opt_f64("f0"),
            values.opt_f64("t"),
            Unseeded,
        ),
        components: &["short rate", "bond price", "forward rate"],
    })
}

inventory::submit! {
    ProcessDescriptor {
        name: "Hjm",
        category: Category::Interest,
        params: &[
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "r0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.03)), doc: "Initial short rate" },
            ParamSpec { name: "p0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Initial bond price" },
            ParamSpec { name: "f0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.03)), doc: "Initial forward rate" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_hjm,
    }
}

fn build_heston2d(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(MultiDim {
        process: Heston2D::<f64>::new(
            [values.opt_f64("x0_1"), values.opt_f64("x0_2")],
            [values.opt_f64("v0_1"), values.opt_f64("v0_2")],
            [values.f64("mu_1"), values.f64("mu_2")],
            [values.f64("theta_1"), values.f64("theta_2")],
            [values.f64("kappa_1"), values.f64("kappa_2")],
            [values.f64("sigma_1"), values.f64("sigma_2")],
            [
                values.f64("rho_1"),
                values.f64("rho_2"),
                values.f64("rho_3"),
                values.f64("rho_4"),
                values.f64("rho_5"),
                values.f64("rho_6"),
            ],
            values.usize("n"),
            values.opt_f64("t"),
            values.opt_bool("use_sym"),
            Unseeded,
        ),
        components: &["asset 1", "asset 2", "variance 1", "variance 2"],
    })
}

inventory::submit! {
    ProcessDescriptor {
        name: "Heston2D",
        category: Category::Volatility,
        params: &[
            ParamSpec { name: "x0_1", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(100.0)), doc: "Asset 1 spot" },
            ParamSpec { name: "x0_2", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(100.0)), doc: "Asset 2 spot" },
            ParamSpec { name: "v0_1", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.04)), doc: "Variance 1 start" },
            ParamSpec { name: "v0_2", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.04)), doc: "Variance 2 start" },
            ParamSpec { name: "mu_1", kind: ParamKind::F64, default: ParamDefault::F64(0.05), doc: "Asset 1 drift" },
            ParamSpec { name: "mu_2", kind: ParamKind::F64, default: ParamDefault::F64(0.05), doc: "Asset 2 drift" },
            ParamSpec { name: "theta_1", kind: ParamKind::F64, default: ParamDefault::F64(0.04), doc: "Variance 1 mean" },
            ParamSpec { name: "theta_2", kind: ParamKind::F64, default: ParamDefault::F64(0.04), doc: "Variance 2 mean" },
            ParamSpec { name: "kappa_1", kind: ParamKind::F64, default: ParamDefault::F64(1.5), doc: "Variance 1 reversion" },
            ParamSpec { name: "kappa_2", kind: ParamKind::F64, default: ParamDefault::F64(1.5), doc: "Variance 2 reversion" },
            ParamSpec { name: "sigma_1", kind: ParamKind::F64, default: ParamDefault::F64(0.3), doc: "Vol of vol 1" },
            ParamSpec { name: "sigma_2", kind: ParamKind::F64, default: ParamDefault::F64(0.3), doc: "Vol of vol 2" },
            ParamSpec { name: "rho_1", kind: ParamKind::F64, default: ParamDefault::F64(-0.5), doc: "Correlation entry 1" },
            ParamSpec { name: "rho_2", kind: ParamKind::F64, default: ParamDefault::F64(0.3), doc: "Correlation entry 2" },
            ParamSpec { name: "rho_3", kind: ParamKind::F64, default: ParamDefault::F64(0.0), doc: "Correlation entry 3" },
            ParamSpec { name: "rho_4", kind: ParamKind::F64, default: ParamDefault::F64(-0.5), doc: "Correlation entry 4" },
            ParamSpec { name: "rho_5", kind: ParamKind::F64, default: ParamDefault::F64(0.0), doc: "Correlation entry 5" },
            ParamSpec { name: "rho_6", kind: ParamKind::F64, default: ParamDefault::F64(0.3), doc: "Correlation entry 6" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
            ParamSpec { name: "use_sym", kind: ParamKind::OptBool, default: ParamDefault::OptBool(Some(true)), doc: "Symmetrise" },
        ],
        build: build_heston2d,
    }
}

fn build_regime_switching(values: &ParamValues) -> Box<dyn ChartSource> {
    let vols = Array1::from_vec(values.f64vec("vols"));
    let dim = vols.len().max(2);
    Box::new(MultiDim {
        process: RegimeSwitchingDiffusion::<f64>::new(
            values.f64("mu"),
            square_matrix(values, "q_matrix", dim),
            vols,
            values.usize("initial_state"),
            values.usize("n"),
            values.opt_f64("s0"),
            values.opt_f64("t"),
            Unseeded,
        ),
        components: &["price", "state"],
    })
}

inventory::submit! {
    ProcessDescriptor {
        name: "RegimeSwitchingDiffusion",
        category: Category::Diffusion,
        params: &[
            ParamSpec { name: "mu", kind: ParamKind::F64, default: ParamDefault::F64(0.05), doc: "Drift" },
            ParamSpec { name: "q_matrix", kind: ParamKind::F64Vec, default: ParamDefault::F64Vec(&[-0.5, 0.5, 0.5, -0.5]), doc: "Generator matrix, row-major" },
            ParamSpec { name: "vols", kind: ParamKind::F64Vec, default: ParamDefault::F64Vec(&[0.1, 0.3]), doc: "Per-state vols" },
            ParamSpec { name: "initial_state", kind: ParamKind::Usize, default: ParamDefault::Usize(0), doc: "Starting regime" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "s0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(100.0)), doc: "Initial spot" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_regime_switching,
    }
}

fn build_multifactor_sabr(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(MultiDim {
        process: MultifactorSabr::<f64>::new(
            values.opt_f64("f0"),
            values.opt_f64("v0"),
            values.f64vec("knots"),
            values.f64vec("beta"),
            values.f64vec("rho"),
            values.f64vec("nu"),
            values.usize("n"),
            values.opt_f64("t"),
            Unseeded,
        ),
        components: &["forward", "variance"],
    })
}

inventory::submit! {
    ProcessDescriptor {
        name: "MultifactorSabr",
        category: Category::Volatility,
        params: &[
            ParamSpec { name: "f0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.03)), doc: "Initial forward" },
            ParamSpec { name: "v0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.2)), doc: "Initial vol" },
            ParamSpec { name: "knots", kind: ParamKind::F64Vec, default: ParamDefault::F64Vec(&[0.5]), doc: "Time knots (buckets = knots+1)" },
            ParamSpec { name: "beta", kind: ParamKind::F64Vec, default: ParamDefault::F64Vec(&[0.5, 0.5]), doc: "Per-factor beta" },
            ParamSpec { name: "rho", kind: ParamKind::F64Vec, default: ParamDefault::F64Vec(&[-0.3, -0.3]), doc: "Per-factor correlation" },
            ParamSpec { name: "nu", kind: ParamKind::F64Vec, default: ParamDefault::F64Vec(&[0.3, 0.3]), doc: "Per-factor vol of vol" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_multifactor_sabr,
    }
}

fn build_fgn(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(Path1D(Fgn::<f64>::new(
        values.f64("hurst"),
        values.usize("n"),
        values.opt_f64("t"),
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "Fgn",
        category: Category::Noise,
        params: &[
            ParamSpec { name: "hurst", kind: ParamKind::F64, default: ParamDefault::F64(0.7), doc: "Hurst exponent" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_fgn,
    }
}

fn build_bilateral_gamma_motion(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(Path1D(BilateralGammaMotion::<f64>::new(
        values.f64("sigma"),
        values.f64("alpha_p"),
        values.f64("lambda_p"),
        values.f64("alpha_m"),
        values.f64("lambda_m"),
        values.usize("n"),
        values.opt_f64("x0"),
        values.opt_f64("t"),
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "BilateralGammaMotion",
        category: Category::Jump,
        params: &[
            ParamSpec { name: "sigma", kind: ParamKind::F64, default: ParamDefault::F64(0.2), doc: "Diffusion scale" },
            ParamSpec { name: "alpha_p", kind: ParamKind::F64, default: ParamDefault::F64(1.0), doc: "Up shape" },
            ParamSpec { name: "lambda_p", kind: ParamKind::F64, default: ParamDefault::F64(10.0), doc: "Up rate" },
            ParamSpec { name: "alpha_m", kind: ParamKind::F64, default: ParamDefault::F64(1.0), doc: "Down shape" },
            ParamSpec { name: "lambda_m", kind: ParamKind::F64, default: ParamDefault::F64(10.0), doc: "Down rate" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "x0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.0)), doc: "Initial value" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_bilateral_gamma_motion,
    }
}

fn build_volterra(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(Path1D(Volterra::<f64>::new(
        VolterraKernel::FractionalBM {
            h: values.f64("hurst"),
        },
        values.usize("n"),
        values.opt_f64("t"),
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "Volterra",
        category: Category::Process,
        params: &[
            ParamSpec { name: "hurst", kind: ParamKind::F64, default: ParamDefault::F64(0.3), doc: "Kernel Hurst" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_volterra,
    }
}

fn build_custom_jt(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(Path1D(CustomJt::new(
        Some(values.usize("n")),
        values.opt_f64("t"),
        Exp::new(values.f64("rate").max(0.01)).unwrap(),
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "CustomJt",
        category: Category::Process,
        params: &[
            ParamSpec { name: "rate", kind: ParamKind::F64, default: ParamDefault::F64(20.0), doc: "Inter-arrival rate" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_custom_jt,
    }
}

fn build_ctrw(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(Path1D(Ctrw::<f64>::new(
        CtrwWaitingLaw::Exponential {
            rate: values.f64("rate"),
        },
        CtrwJumpLaw::Normal {
            mean: values.f64("jump_mean"),
            std: values.f64("jump_std"),
        },
        values.usize("n"),
        values.opt_f64("x0"),
        values.opt_f64("t"),
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "Ctrw",
        category: Category::Process,
        params: &[
            ParamSpec { name: "rate", kind: ParamKind::F64, default: ParamDefault::F64(20.0), doc: "Waiting-time rate" },
            ParamSpec { name: "jump_mean", kind: ParamKind::F64, default: ParamDefault::F64(0.0), doc: "Mean jump" },
            ParamSpec { name: "jump_std", kind: ParamKind::F64, default: ParamDefault::F64(0.1), doc: "Jump std" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "x0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(0.0)), doc: "Initial value" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_ctrw,
    }
}

fn build_compound_custom(values: &ParamValues) -> Box<dyn ChartSource> {
    let n = values.usize("n");
    let t = values.opt_f64("t");
    let times = Exp::new(values.f64("rate").max(0.01)).unwrap();
    let customjt = CustomJt::new(
        Some(n),
        t,
        Exp::new(values.f64("rate").max(0.01)).unwrap(),
        Unseeded,
    );
    Box::new(MultiDim {
        process: CompoundCustom::new(
            Some(n),
            t,
            Normal::new(values.f64("jump_mean"), values.f64("jump_std").max(1e-9)).unwrap(),
            times,
            customjt,
            Unseeded,
        ),
        components: &["events", "sum", "compensated"],
    })
}

inventory::submit! {
    ProcessDescriptor {
        name: "CompoundCustom",
        category: Category::Process,
        params: &[
            ParamSpec { name: "rate", kind: ParamKind::F64, default: ParamDefault::F64(20.0), doc: "Inter-arrival rate" },
            ParamSpec { name: "jump_mean", kind: ParamKind::F64, default: ParamDefault::F64(0.0), doc: "Mean jump size" },
            ParamSpec { name: "jump_std", kind: ParamKind::F64, default: ParamDefault::F64(0.1), doc: "Jump size std" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_compound_custom,
    }
}

fn build_compound_poisson(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(MultiDim {
        process: CompoundPoisson::new(
            Normal::new(values.f64("jump_mean"), values.f64("jump_std").max(1e-9)).unwrap(),
            Poisson::new(
                values.f64("lambda"),
                Some(values.usize("n")),
                values.opt_f64("t"),
                Unseeded,
            ),
            Unseeded,
        ),
        components: &["cumulative", "jump sizes", "arrival times"],
    })
}

inventory::submit! {
    ProcessDescriptor {
        name: "CompoundPoisson",
        category: Category::Process,
        params: &[
            ParamSpec { name: "lambda", kind: ParamKind::F64, default: ParamDefault::F64(5.0), doc: "Jump intensity" },
            ParamSpec { name: "jump_mean", kind: ParamKind::F64, default: ParamDefault::F64(0.0), doc: "Mean jump size" },
            ParamSpec { name: "jump_std", kind: ParamKind::F64, default: ParamDefault::F64(0.2), doc: "Jump size std" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_compound_poisson,
    }
}

fn build_multifactor_heston(values: &ParamValues) -> Box<dyn ChartSource> {
    Box::new(TupleDim(MultifactorHeston::<f64, 2>::new(
        values.opt_f64("s0"),
        [values.f64("v0_1"), values.f64("v0_2")],
        [values.f64("kappa_1"), values.f64("kappa_2")],
        [values.f64("theta_1"), values.f64("theta_2")],
        [values.f64("sigma_1"), values.f64("sigma_2")],
        [values.f64("rho_1"), values.f64("rho_2")],
        values.f64("mu"),
        values.usize("n"),
        values.opt_f64("t"),
        Unseeded,
    )))
}

inventory::submit! {
    ProcessDescriptor {
        name: "MultifactorHeston",
        category: Category::Volatility,
        params: &[
            ParamSpec { name: "s0", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(100.0)), doc: "Initial spot" },
            ParamSpec { name: "v0_1", kind: ParamKind::F64, default: ParamDefault::F64(0.04), doc: "Variance 1 start" },
            ParamSpec { name: "v0_2", kind: ParamKind::F64, default: ParamDefault::F64(0.04), doc: "Variance 2 start" },
            ParamSpec { name: "kappa_1", kind: ParamKind::F64, default: ParamDefault::F64(1.5), doc: "Variance 1 reversion" },
            ParamSpec { name: "kappa_2", kind: ParamKind::F64, default: ParamDefault::F64(0.5), doc: "Variance 2 reversion" },
            ParamSpec { name: "theta_1", kind: ParamKind::F64, default: ParamDefault::F64(0.04), doc: "Variance 1 mean" },
            ParamSpec { name: "theta_2", kind: ParamKind::F64, default: ParamDefault::F64(0.04), doc: "Variance 2 mean" },
            ParamSpec { name: "sigma_1", kind: ParamKind::F64, default: ParamDefault::F64(0.3), doc: "Vol of vol 1" },
            ParamSpec { name: "sigma_2", kind: ParamKind::F64, default: ParamDefault::F64(0.2), doc: "Vol of vol 2" },
            ParamSpec { name: "rho_1", kind: ParamKind::F64, default: ParamDefault::F64(-0.7), doc: "Correlation 1" },
            ParamSpec { name: "rho_2", kind: ParamKind::F64, default: ParamDefault::F64(-0.5), doc: "Correlation 2" },
            ParamSpec { name: "mu", kind: ParamKind::F64, default: ParamDefault::F64(0.05), doc: "Drift" },
            ParamSpec { name: "n", kind: ParamKind::Usize, default: ParamDefault::Usize(1000), doc: "Steps" },
            ParamSpec { name: "t", kind: ParamKind::OptF64, default: ParamDefault::OptF64(Some(1.0)), doc: "Horizon" },
        ],
        build: build_multifactor_heston,
    }
}
