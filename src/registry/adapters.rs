//! Output adapters mapping heterogeneous process outputs onto [`ChartSource`].
//!
//! Each adapter wraps a concrete process and converts its sampled output into
//! labelled series with the simulation step index on the x-axis (uniform across
//! processes, since not every model carries a time horizon).

use ndarray::Array1;
use stochastic_rs_stochastic::ProcessExt;

use crate::registry::ChartSource;
use crate::registry::NamedSeries;

fn series(arr: &Array1<f64>, label: String) -> NamedSeries {
    let points = arr.iter().enumerate().map(|(i, &y)| (i as f64, y)).collect();
    NamedSeries { label, points }
}

/// Adapter for scalar-path processes (`Output = Array1<f64>`).
pub struct Path1D<P>(pub P);

impl<P> ChartSource for Path1D<P>
where
    P: ProcessExt<f64, Output = Array1<f64>>,
{
    fn sample_par(&self, m: usize) -> Vec<Vec<NamedSeries>> {
        self.0
            .sample_par(m)
            .iter()
            .enumerate()
            .map(|(i, path)| vec![series(path, format!("#{i}"))])
            .collect()
    }
}

/// Adapter for `N`-state processes (`Output = [Array1<f64>; N]`), e.g. the
/// asset/variance pair of stochastic-volatility models.
pub struct MultiDim<P>(pub P);

impl<const N: usize, P> ChartSource for MultiDim<P>
where
    P: ProcessExt<f64, Output = [Array1<f64>; N]>,
{
    fn sample_par(&self, m: usize) -> Vec<Vec<NamedSeries>> {
        self.0
            .sample_par(m)
            .iter()
            .enumerate()
            .map(|(i, components)| {
                components
                    .iter()
                    .enumerate()
                    .map(|(k, comp)| series(comp, format!("#{i}.{k}")))
                    .collect()
            })
            .collect()
    }
}
