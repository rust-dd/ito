//! Noise process registrations.

use crate::process;
use stochastic_rs_stochastic::noise::cfgns::Cfgns;
use stochastic_rs_stochastic::noise::cgns::Cgns;
use stochastic_rs_stochastic::noise::gn::Gn;
use stochastic_rs_stochastic::noise::wn::Wn;

process! {
    name: "Cfgns",
    ty: Cfgns<f64>,
    category: Noise,
    output: MultiDim,
    components: ["x1", "x2"],
    params: [
        hurst : f64     = 0.4 ; "Hurst exponent",
        rho   : f64     = -0.5 ; "Correlation",
        n     : usize   = 1000 ; "Steps",
        t     : opt_f64 = Some(1.0) ; "Horizon",
    ],
}

process! {
    name: "Cgns",
    ty: Cgns<f64>,
    category: Noise,
    output: MultiDim,
    components: ["x1", "x2"],
    params: [
        rho : f64     = -0.5 ; "Correlation",
        n   : usize   = 1000 ; "Steps",
        t   : opt_f64 = Some(1.0) ; "Horizon",
    ],
}

process! {
    name: "Gn",
    ty: Gn<f64>,
    category: Noise,
    output: Path1D,
    components: [],
    params: [
        n : usize   = 1000 ; "Steps",
        t : opt_f64 = Some(1.0) ; "Horizon",
    ],
}

process! {
    name: "Wn",
    ty: Wn<f64>,
    category: Noise,
    output: Path1D,
    components: [],
    params: [
        n       : usize   = 1000 ; "Steps",
        mean    : opt_f64 = Some(0.5) ; "mean",
        std_dev : opt_f64 = Some(0.5) ; "std_dev",
    ],
}
