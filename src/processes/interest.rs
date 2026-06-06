//! Interest process registrations.

use crate::process;
use stochastic_rs_stochastic::interest::bgm::Bgm;
use stochastic_rs_stochastic::interest::duffie_kan::DuffieKan;
use stochastic_rs_stochastic::interest::duffie_kan_jump_exp::DuffieKanJumpExp;
use stochastic_rs_stochastic::interest::fractional_vasicek::FVasicek;
use stochastic_rs_stochastic::interest::lmm::Lmm;
use stochastic_rs_stochastic::interest::vasicek::Vasicek;
use stochastic_rs_stochastic::interest::wu_zhang::WuZhangD;

process! {
    name: "Bgm",
    ty: Bgm<f64>,
    category: Interest,
    output: Curve,
    components: [],
    params: [
        lambda : f64vec  = &[0.5, 0.3] ; "Jump intensity",
        x0     : f64vec  = &[0.5, 0.3] ; "Initial value",
        xn     : usize   = 2 ; "xn",
        t      : opt_f64 = Some(1.0) ; "Horizon",
        n      : usize   = 1000 ; "Steps",
    ],
}

process! {
    name: "DuffieKan",
    ty: DuffieKan<f64>,
    category: Interest,
    output: MultiDim,
    components: ["x1", "x2"],
    params: [
        alpha  : f64     = 0.5 ; "alpha",
        beta   : f64     = 0.5 ; "beta",
        gamma  : f64     = 0.5 ; "gamma",
        rho    : f64     = -0.5 ; "Correlation",
        a1     : f64     = 0.5 ; "a1",
        b1     : f64     = 0.5 ; "b1",
        c1     : f64     = 0.5 ; "c1",
        sigma1 : f64     = 0.2 ; "sigma1",
        a2     : f64     = 0.5 ; "a2",
        b2     : f64     = 0.5 ; "b2",
        c2     : f64     = 0.5 ; "c2",
        sigma2 : f64     = 0.2 ; "sigma2",
        n      : usize   = 1000 ; "Steps",
        r0     : opt_f64 = Some(0.5) ; "r0",
        x0     : opt_f64 = Some(0.5) ; "Initial value",
        t      : opt_f64 = Some(1.0) ; "Horizon",
    ],
}

process! {
    name: "DuffieKanJumpExp",
    ty: DuffieKanJumpExp<f64>,
    category: Interest,
    output: MultiDim,
    components: ["x1", "x2"],
    params: [
        alpha      : f64     = 0.5 ; "alpha",
        beta       : f64     = 0.5 ; "beta",
        gamma      : f64     = 0.5 ; "gamma",
        rho        : f64     = -0.5 ; "Correlation",
        a1         : f64     = 0.5 ; "a1",
        b1         : f64     = 0.5 ; "b1",
        c1         : f64     = 0.5 ; "c1",
        sigma1     : f64     = 0.2 ; "sigma1",
        a2         : f64     = 0.5 ; "a2",
        b2         : f64     = 0.5 ; "b2",
        c2         : f64     = 0.5 ; "c2",
        sigma2     : f64     = 0.2 ; "sigma2",
        lambda     : f64     = 1.0 ; "Jump intensity",
        jump_scale : f64     = 0.5 ; "jump_scale",
        n          : usize   = 1000 ; "Steps",
        r0         : opt_f64 = Some(0.5) ; "r0",
        x0         : opt_f64 = Some(0.5) ; "Initial value",
        t          : opt_f64 = Some(1.0) ; "Horizon",
    ],
}

process! {
    name: "FVasicek",
    ty: FVasicek<f64>,
    category: Interest,
    output: Path1D,
    components: [],
    params: [
        hurst : f64     = 0.4 ; "Hurst exponent",
        theta : f64     = 0.5 ; "Mean / reversion",
        mu    : f64     = 0.1 ; "Drift / mean",
        sigma : f64     = 0.2 ; "Diffusion scale",
        n     : usize   = 1000 ; "Steps",
        x0    : opt_f64 = Some(0.5) ; "Initial value",
        t     : opt_f64 = Some(1.0) ; "Horizon",
    ],
}

process! {
    name: "Lmm",
    ty: Lmm<f64>,
    category: Interest,
    output: Curve,
    components: [],
    params: [
        tenor : f64vec  = &[0.5, 1.0, 1.5] ; "tenor",
        l0    : f64vec  = &[0.03, 0.03] ; "l0",
        sigma : f64vec  = &[0.2, 0.2] ; "Diffusion scale",
        n     : usize   = 1000 ; "Steps",
        t     : opt_f64 = Some(1.0) ; "Horizon",
    ],
}

process! {
    name: "Vasicek",
    ty: Vasicek<f64>,
    category: Interest,
    output: Path1D,
    components: [],
    params: [
        theta : f64     = 0.5 ; "Mean / reversion",
        mu    : f64     = 0.1 ; "Drift / mean",
        sigma : f64     = 0.2 ; "Diffusion scale",
        n     : usize   = 1000 ; "Steps",
        x0    : opt_f64 = Some(0.5) ; "Initial value",
        t     : opt_f64 = Some(1.0) ; "Horizon",
    ],
}

process! {
    name: "WuZhangD",
    ty: WuZhangD<f64>,
    category: Interest,
    output: Curve,
    components: [],
    params: [
        alpha  : f64vec  = &[0.3, 0.2] ; "alpha",
        beta   : f64vec  = &[0.3, 0.2] ; "beta",
        nu     : f64vec  = &[0.5, 0.3] ; "nu",
        lambda : f64vec  = &[0.5, 0.3] ; "Jump intensity",
        x0     : f64vec  = &[0.5, 0.3] ; "Initial value",
        v0     : f64vec  = &[0.5, 0.3] ; "Initial variance",
        xn     : usize   = 2 ; "xn",
        t      : opt_f64 = Some(1.0) ; "Horizon",
        n      : usize   = 1000 ; "Steps",
    ],
}
