mod registry;
mod processes;

use registry::ParamValues;
use registry::registry;

fn main() {
    let reg = registry();
    println!("{} processes registered", reg.len());
    for d in &reg {
        println!(
            "[{}] {} ({} params)",
            d.category.label(),
            d.name,
            d.params.len()
        );
    }

    if let Some(d) = reg.iter().find(|d| d.name == "Gbm") {
        let values = ParamValues::from_defaults(d.params);
        let src = (d.build)(&values);
        let samples = src.sample_par(2);
        println!(
            "Gbm -> {} samples, first has {} points",
            samples.len(),
            samples[0][0].points.len()
        );
    }
}
