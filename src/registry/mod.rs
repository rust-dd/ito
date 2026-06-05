//! Runtime metadata and type-erased builders for `stochastic-rs` processes.
//!
//! Rust has no reflection, so each process is described explicitly (via the
//! [`process!`](crate::process) macro) with a [`ProcessDescriptor`]: its name,
//! category, the ordered [`ParamSpec`] list mirroring its `new(...)` arguments,
//! and a `build` function that constructs it from runtime [`ParamValues`] and
//! erases it behind [`ChartSource`]. Descriptors self-register through
//! `inventory`; [`registry`] returns them sorted.

pub mod adapters;
mod macros;

use std::collections::HashMap;

/// One plottable trajectory (or one component of a multi-state sample).
#[derive(Clone)]
pub struct NamedSeries {
    pub label: String,
    pub points: Vec<(f64, f64)>,
}

/// The type of a constructor parameter; drives editing and value coercion.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    F64,
    Usize,
    OptF64,
    OptBool,
}

impl ParamKind {
    /// Short hint shown next to the field in the TUI.
    pub fn hint(self) -> &'static str {
        match self {
            ParamKind::F64 => "f64",
            ParamKind::Usize => "uint",
            ParamKind::OptF64 => "f64?",
            ParamKind::OptBool => "bool?",
        }
    }
}

/// Compile-time default for a parameter, used to seed the form.
#[derive(Clone, Copy)]
pub enum ParamDefault {
    F64(f64),
    Usize(usize),
    OptF64(Option<f64>),
    OptBool(Option<bool>),
}

/// Static description of a single constructor argument.
pub struct ParamSpec {
    pub name: &'static str,
    pub kind: ParamKind,
    pub default: ParamDefault,
    pub doc: &'static str,
}

/// Grouping for the process list. Declaration order is the display order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Category {
    Diffusion,
    Volatility,
    Jump,
    Interest,
    Autoregressive,
    Rough,
    Noise,
    Correlation,
    Process,
}

impl Category {
    pub fn label(self) -> &'static str {
        match self {
            Category::Diffusion => "Diffusion",
            Category::Volatility => "Volatility",
            Category::Jump => "Jump",
            Category::Interest => "Interest",
            Category::Autoregressive => "Autoregressive",
            Category::Rough => "Rough",
            Category::Noise => "Noise",
            Category::Correlation => "Correlation",
            Category::Process => "Process",
        }
    }
}

/// A single runtime parameter value entered by the user.
#[derive(Clone, Copy)]
pub enum ParamValue {
    F64(f64),
    Usize(usize),
    OptF64(Option<f64>),
    OptBool(Option<bool>),
}

/// Runtime parameter values keyed by parameter name, consumed by `build`.
#[derive(Default, Clone)]
pub struct ParamValues {
    map: HashMap<String, ParamValue>,
}

impl ParamValues {
    pub fn set(&mut self, name: &str, value: ParamValue) {
        self.map.insert(name.to_string(), value);
    }

    pub fn f64(&self, name: &str) -> f64 {
        match self.map.get(name) {
            Some(ParamValue::F64(v)) => *v,
            _ => 0.0,
        }
    }

    pub fn usize(&self, name: &str) -> usize {
        match self.map.get(name) {
            Some(ParamValue::Usize(v)) => *v,
            _ => 0,
        }
    }

    pub fn opt_f64(&self, name: &str) -> Option<f64> {
        match self.map.get(name) {
            Some(ParamValue::OptF64(v)) => *v,
            _ => None,
        }
    }

    pub fn opt_bool(&self, name: &str) -> Option<bool> {
        match self.map.get(name) {
            Some(ParamValue::OptBool(v)) => *v,
            _ => None,
        }
    }
}

/// A constructed, ready-to-sample process erased behind a uniform interface.
pub trait ChartSource: Send + Sync {
    /// Draw `m` independent samples; each yields one or more labelled series
    /// (1 for a scalar path, `N` for an `N`-state model).
    fn sample_par(&self, m: usize) -> Vec<Vec<NamedSeries>>;
}

/// Static, self-registered description of one process.
pub struct ProcessDescriptor {
    pub name: &'static str,
    pub category: Category,
    pub params: &'static [ParamSpec],
    pub build: fn(&ParamValues) -> Box<dyn ChartSource>,
}

inventory::collect!(ProcessDescriptor);

/// All registered descriptors, sorted by category then name.
pub fn registry() -> Vec<&'static ProcessDescriptor> {
    let mut all: Vec<&'static ProcessDescriptor> = inventory::iter::<ProcessDescriptor>().collect();
    all.sort_by(|a, b| (a.category, a.name).cmp(&(b.category, b.name)));
    all
}
