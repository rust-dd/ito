//! Hand-written registrations for processes whose constructors take an argument
//! the scalar/vector form can't express — here a fixed enum variant. The scalar
//! parameters are still exposed; the non-scalar argument is pinned to its
//! conventional value.

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
