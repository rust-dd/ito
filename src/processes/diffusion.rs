//! Diffusion process registrations (scalar `Array1` paths).

use crate::process;
use stochastic_rs_stochastic::diffusion::cev::Cev;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::feller::FellerLogistic;
use stochastic_rs_stochastic::diffusion::fgbm::Fgbm;
use stochastic_rs_stochastic::diffusion::fou::Fou;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::diffusion::gompertz::Gompertz;
use stochastic_rs_stochastic::diffusion::jacobi::Jacobi;
use stochastic_rs_stochastic::diffusion::logistic::Logistic;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::diffusion::three_half::ThreeHalf;
use stochastic_rs_stochastic::diffusion::verhulst::Verhulst;

process! {
    name: "Gbm",
    ty: Gbm<f64>,
    category: Diffusion,
    output: Path1D,
    params: [
        mu:    f64     = 0.05        ; "Drift",
        sigma: f64     = 0.2         ; "Diffusion scale",
        n:     usize   = 1000        ; "Steps",
        x0:    opt_f64 = Some(100.0) ; "Initial value",
        t:     opt_f64 = Some(1.0)   ; "Horizon",
    ],
}

process! {
    name: "Ou",
    ty: Ou<f64>,
    category: Diffusion,
    output: Path1D,
    params: [
        theta: f64     = 1.0       ; "Mean-reversion speed",
        mu:    f64     = 1.2       ; "Long-run mean",
        sigma: f64     = 0.3       ; "Diffusion scale",
        n:     usize   = 1000      ; "Steps",
        x0:    opt_f64 = Some(0.5) ; "Initial value",
        t:     opt_f64 = Some(1.0) ; "Horizon",
    ],
}

process! {
    name: "Cir",
    ty: Cir<f64>,
    category: Diffusion,
    output: Path1D,
    params: [
        theta:   f64      = 2.0        ; "Mean-reversion speed",
        mu:      f64      = 0.5        ; "Long-run mean",
        sigma:   f64      = 0.2        ; "Diffusion scale",
        n:       usize    = 1000       ; "Steps",
        x0:      opt_f64  = Some(0.5)  ; "Initial value",
        t:       opt_f64  = Some(1.0)  ; "Horizon",
        use_sym: opt_bool = Some(true) ; "Symmetrise to keep positive",
    ],
}

process! {
    name: "Fou",
    ty: Fou<f64>,
    category: Diffusion,
    output: Path1D,
    params: [
        hurst: f64     = 0.7       ; "Hurst exponent",
        theta: f64     = 1.0       ; "Mean-reversion speed",
        mu:    f64     = 1.0       ; "Long-run mean",
        sigma: f64     = 0.3       ; "Diffusion scale",
        n:     usize   = 1000      ; "Steps",
        x0:    opt_f64 = Some(0.5) ; "Initial value",
        t:     opt_f64 = Some(1.0) ; "Horizon",
    ],
}

process! {
    name: "Fgbm",
    ty: Fgbm<f64>,
    category: Diffusion,
    output: Path1D,
    params: [
        hurst: f64     = 0.7         ; "Hurst exponent",
        mu:    f64     = 0.05        ; "Drift",
        sigma: f64     = 0.2         ; "Diffusion scale",
        n:     usize   = 1000        ; "Steps",
        x0:    opt_f64 = Some(100.0) ; "Initial value",
        t:     opt_f64 = Some(1.0)   ; "Horizon",
    ],
}

process! {
    name: "Jacobi",
    ty: Jacobi<f64>,
    category: Diffusion,
    output: Path1D,
    params: [
        alpha: f64     = 0.5       ; "Lower pull",
        beta:  f64     = 0.5       ; "Upper pull",
        sigma: f64     = 0.3       ; "Diffusion scale",
        n:     usize   = 1000      ; "Steps",
        x0:    opt_f64 = Some(0.5) ; "Initial value in (0,1)",
        t:     opt_f64 = Some(1.0) ; "Horizon",
    ],
}

process! {
    name: "Gompertz",
    ty: Gompertz<f64>,
    category: Diffusion,
    output: Path1D,
    params: [
        a:     f64     = 1.0       ; "Growth rate",
        b:     f64     = 0.5       ; "Decay rate",
        sigma: f64     = 0.2       ; "Diffusion scale",
        n:     usize   = 1000      ; "Steps",
        x0:    opt_f64 = Some(0.5) ; "Initial value",
        t:     opt_f64 = Some(1.0) ; "Horizon",
    ],
}

process! {
    name: "ThreeHalf",
    ty: ThreeHalf<f64>,
    category: Diffusion,
    output: Path1D,
    params: [
        kappa: f64     = 2.0       ; "Mean-reversion speed",
        mu:    f64     = 0.5       ; "Long-run mean",
        sigma: f64     = 0.2       ; "Diffusion scale",
        n:     usize   = 1000      ; "Steps",
        x0:    opt_f64 = Some(0.5) ; "Initial value",
        t:     opt_f64 = Some(1.0) ; "Horizon",
    ],
}

process! {
    name: "Verhulst",
    ty: Verhulst<f64>,
    category: Diffusion,
    output: Path1D,
    params: [
        r:     f64      = 1.5        ; "Growth rate",
        k:     f64      = 1.0        ; "Carrying capacity",
        sigma: f64      = 0.2        ; "Diffusion scale",
        n:     usize    = 1000       ; "Steps",
        x0:    opt_f64  = Some(0.3)  ; "Initial value",
        t:     opt_f64  = Some(1.0)  ; "Horizon",
        clamp: opt_bool = Some(true) ; "Clamp to keep positive",
    ],
}

process! {
    name: "Logistic",
    ty: Logistic<f64>,
    category: Diffusion,
    output: Path1D,
    params: [
        a:  f64     = 1.5       ; "Growth rate",
        b:  f64     = 1.0       ; "Saturation rate",
        n:  usize   = 1000      ; "Steps",
        x0: opt_f64 = Some(0.3) ; "Initial value",
        t:  opt_f64 = Some(1.0) ; "Horizon",
    ],
}

process! {
    name: "Cev",
    ty: Cev<f64>,
    category: Diffusion,
    output: Path1D,
    params: [
        mu:    f64     = 0.05        ; "Drift",
        sigma: f64     = 0.3         ; "Diffusion scale",
        gamma: f64     = 0.8         ; "Elasticity",
        n:     usize   = 1000        ; "Steps",
        x0:    opt_f64 = Some(100.0) ; "Initial value",
        t:     opt_f64 = Some(1.0)   ; "Horizon",
    ],
}

process! {
    name: "FellerLogistic",
    ty: FellerLogistic<f64>,
    category: Diffusion,
    output: Path1D,
    params: [
        kappa:   f64      = 2.0        ; "Mean-reversion speed",
        theta:   f64      = 0.5        ; "Long-run mean",
        sigma:   f64      = 0.2        ; "Diffusion scale",
        n:       usize    = 1000       ; "Steps",
        x0:      opt_f64  = Some(0.5)  ; "Initial value",
        t:       opt_f64  = Some(1.0)  ; "Horizon",
        use_sym: opt_bool = Some(true) ; "Symmetrise to keep positive",
    ],
}
