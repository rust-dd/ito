//! Process registrations grouped by category. Each submodule populates the
//! global registry through the `process!` macro.

pub mod autoregressive;
pub mod correlation;
pub mod diffusion;
pub mod interest;
pub mod jump;
pub mod rough;
pub mod volatility;
