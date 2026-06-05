//! Hand-written registrations for processes whose constructors take an argument
//! the scalar/vector form can't express — here a fixed enum variant. The scalar
//! parameters are still exposed; the non-scalar argument is pinned to its
//! conventional value.

use rand_distr::Normal;
use stochastic_rs_stochastic::correlation::transformed_ou::TransformedOU;
use stochastic_rs_stochastic::correlation::transformed_ou::Transformation;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::process::cpoisson::CompoundPoisson;
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
use crate::registry::adapters::MultiDim;
use crate::registry::adapters::Path1D;

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
