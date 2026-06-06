//! Autoregressive process registrations.

use crate::process;
use stochastic_rs_stochastic::autoregressive::agrach::Agarch;
use stochastic_rs_stochastic::autoregressive::ar::ARp;
use stochastic_rs_stochastic::autoregressive::arch::Arch;
use stochastic_rs_stochastic::autoregressive::arima::Arima;
use stochastic_rs_stochastic::autoregressive::egarch::Egarch;
use stochastic_rs_stochastic::autoregressive::garch::Garch;
use stochastic_rs_stochastic::autoregressive::ma::MAq;
use stochastic_rs_stochastic::autoregressive::sarima::Sarima;
use stochastic_rs_stochastic::autoregressive::tgarch::Tgarch;

process! {
    name: "Agarch",
    ty: Agarch<f64>,
    category: Autoregressive,
    output: Path1D,
    components: [],
    params: [
        omega : f64    = 0.2 ; "omega",
        alpha : f64vec = &[0.1] ; "alpha",
        delta : f64vec = &[0.05] ; "delta",
        beta  : f64vec = &[0.85] ; "beta",
        n     : usize  = 1000 ; "Steps",
    ],
}

process! {
    name: "ARp",
    ty: ARp<f64>,
    category: Autoregressive,
    output: Path1D,
    components: [],
    params: [
        phi   : f64vec     = &[0.5] ; "phi",
        sigma : f64        = 0.2 ; "Diffusion scale",
        n     : usize      = 1000 ; "Steps",
        x0    : opt_f64vec = None ; "Initial value",
    ],
}

process! {
    name: "Arch",
    ty: Arch<f64>,
    category: Autoregressive,
    output: Path1D,
    components: [],
    params: [
        omega : f64    = 0.2 ; "omega",
        alpha : f64vec = &[0.1] ; "alpha",
        n     : usize  = 1000 ; "Steps",
    ],
}

process! {
    name: "Arima",
    ty: Arima<f64>,
    category: Autoregressive,
    output: Path1D,
    components: [],
    params: [
        ar_coefs : f64vec = &[0.5, 0.3] ; "ar_coefs",
        ma_coefs : f64vec = &[0.5, 0.3] ; "ma_coefs",
        d        : usize  = 1 ; "d",
        sigma    : f64    = 0.2 ; "Diffusion scale",
        n        : usize  = 1000 ; "Steps",
    ],
}

process! {
    name: "Egarch",
    ty: Egarch<f64>,
    category: Autoregressive,
    output: Path1D,
    components: [],
    params: [
        omega : f64    = 0.2 ; "omega",
        alpha : f64vec = &[0.1] ; "alpha",
        gamma : f64vec = &[0.05] ; "gamma",
        beta  : f64vec = &[0.85] ; "beta",
        n     : usize  = 1000 ; "Steps",
    ],
}

process! {
    name: "Garch",
    ty: Garch<f64>,
    category: Autoregressive,
    output: Path1D,
    components: [],
    params: [
        omega : f64    = 0.2 ; "omega",
        alpha : f64vec = &[0.1] ; "alpha",
        beta  : f64vec = &[0.85] ; "beta",
        n     : usize  = 1000 ; "Steps",
    ],
}

process! {
    name: "MAq",
    ty: MAq<f64>,
    category: Autoregressive,
    output: Path1D,
    components: [],
    params: [
        theta : f64vec = &[0.4] ; "Mean / reversion",
        sigma : f64    = 0.2 ; "Diffusion scale",
        n     : usize  = 1000 ; "Steps",
    ],
}

process! {
    name: "Sarima",
    ty: Sarima<f64>,
    category: Autoregressive,
    output: Path1D,
    components: [],
    params: [
        non_seasonal_ar_coefs : f64vec = &[0.5, 0.3] ; "non_seasonal_ar_coefs",
        non_seasonal_ma_coefs : f64vec = &[0.5, 0.3] ; "non_seasonal_ma_coefs",
        seasonal_ar_coefs     : f64vec = &[0.5, 0.3] ; "seasonal_ar_coefs",
        seasonal_ma_coefs     : f64vec = &[0.5, 0.3] ; "seasonal_ma_coefs",
        d                     : usize  = 1 ; "d",
        D                     : usize  = 1 ; "D",
        s                     : usize  = 12 ; "s",
        sigma                 : f64    = 0.2 ; "Diffusion scale",
        n                     : usize  = 1000 ; "Steps",
    ],
}

process! {
    name: "Tgarch",
    ty: Tgarch<f64>,
    category: Autoregressive,
    output: Path1D,
    components: [],
    params: [
        omega : f64    = 0.2 ; "omega",
        alpha : f64vec = &[0.1] ; "alpha",
        gamma : f64vec = &[0.05] ; "gamma",
        beta  : f64vec = &[0.85] ; "beta",
        n     : usize  = 1000 ; "Steps",
    ],
}
