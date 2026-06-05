//! Output adapters mapping heterogeneous process outputs onto [`ChartSource`].
//!
//! Each adapter wraps a concrete process and converts its sampled output into
//! labelled series with the simulation step index on the x-axis (uniform across
//! processes, since not every model carries a time horizon).

use ndarray::Array1;
use num_complex::Complex;
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
            .map(|path| vec![series(path, "path".to_string())])
            .collect()
    }
}

/// Adapter for `N`-state processes (`Output = [Array1<f64>; N]`), e.g. the
/// asset/variance pair of stochastic-volatility models. `components` names the
/// state variables so the chart labels them ("asset", "variance", …) and each
/// can be viewed on its own scale.
pub struct MultiDim<P> {
    pub process: P,
    pub components: &'static [&'static str],
}

impl<const N: usize, P> ChartSource for MultiDim<P>
where
    P: ProcessExt<f64, Output = [Array1<f64>; N]>,
{
    fn sample_par(&self, m: usize) -> Vec<Vec<NamedSeries>> {
        self.process
            .sample_par(m)
            .iter()
            .map(|components| {
                components
                    .iter()
                    .enumerate()
                    .map(|(k, comp)| {
                        let name = self.components.get(k).copied().unwrap_or("comp");
                        series(comp, name.to_string())
                    })
                    .collect()
            })
            .collect()
    }
}

/// Adapter for complex-valued scalar paths (`Output = Array1<Complex<f64>>`),
/// charting the real and imaginary parts as two components.
pub struct ComplexPath<P>(pub P);

impl<P> ChartSource for ComplexPath<P>
where
    P: ProcessExt<f64, Output = Array1<Complex<f64>>>,
{
    fn sample_par(&self, m: usize) -> Vec<Vec<NamedSeries>> {
        self.0
            .sample_par(m)
            .iter()
            .map(|path| {
                let real: Vec<(f64, f64)> =
                    path.iter().enumerate().map(|(i, z)| (i as f64, z.re)).collect();
                let imag: Vec<(f64, f64)> =
                    path.iter().enumerate().map(|(i, z)| (i as f64, z.im)).collect();
                vec![
                    NamedSeries { label: "real".to_string(), points: real },
                    NamedSeries { label: "imag".to_string(), points: imag },
                ]
            })
            .collect()
    }
}
