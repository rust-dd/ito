//! Process registrations grouped by category. Each submodule populates the
//! global registry through the `process!` macro, plus hand-written
//! `manual` registrations for non-scalar constructors.

pub mod autoregressive;
pub mod correlation;
pub mod diffusion;
pub mod interest;
pub mod jump;
pub mod manual;
pub mod noise;
pub mod process;
pub mod rough;
pub mod sheet;
pub mod volatility;
