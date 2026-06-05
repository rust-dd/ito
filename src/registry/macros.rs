//! The `process!` registration macro and its kind-dispatch helpers.
//!
//! `process!` mirrors a real `new(...)` constructor signature and emits a
//! self-registering [`ProcessDescriptor`](crate::registry::ProcessDescriptor):
//! the `ParamSpec` list, a `build` closure that calls the real constructor with
//! `Unseeded` seeding, the chosen output adapter, and the `inventory` entry.
//! Because it compiles against the real `new`, any signature drift is a
//! compile error.

#[macro_export]
macro_rules! ito_param_kind {
    (f64) => { $crate::registry::ParamKind::F64 };
    (usize) => { $crate::registry::ParamKind::Usize };
    (opt_f64) => { $crate::registry::ParamKind::OptF64 };
    (opt_bool) => { $crate::registry::ParamKind::OptBool };
}

#[macro_export]
macro_rules! ito_param_default {
    (f64, $d:expr) => { $crate::registry::ParamDefault::F64($d) };
    (usize, $d:expr) => { $crate::registry::ParamDefault::Usize($d) };
    (opt_f64, $d:expr) => { $crate::registry::ParamDefault::OptF64($d) };
    (opt_bool, $d:expr) => { $crate::registry::ParamDefault::OptBool($d) };
}

#[macro_export]
macro_rules! ito_param_get {
    (f64, $v:expr, $name:expr) => { $v.f64($name) };
    (usize, $v:expr, $name:expr) => { $v.usize($name) };
    (opt_f64, $v:expr, $name:expr) => { $v.opt_f64($name) };
    (opt_bool, $v:expr, $name:expr) => { $v.opt_bool($name) };
}

#[macro_export]
macro_rules! process {
    (
        name: $name:literal,
        ty: $ty:ty,
        category: $cat:ident,
        output: $out:ident,
        params: [ $( $pname:ident : $pkind:ident = $pdef:expr ; $pdoc:literal ),* $(,)? ] $(,)?
    ) => {
        ::inventory::submit! {
            $crate::registry::ProcessDescriptor {
                name: $name,
                category: $crate::registry::Category::$cat,
                params: &[
                    $(
                        $crate::registry::ParamSpec {
                            name: ::core::stringify!($pname),
                            kind: $crate::ito_param_kind!($pkind),
                            default: $crate::ito_param_default!($pkind, $pdef),
                            doc: $pdoc,
                        }
                    ),*
                ],
                build: {
                    fn build(
                        values: &$crate::registry::ParamValues,
                    ) -> ::std::boxed::Box<dyn $crate::registry::ChartSource> {
                        ::std::boxed::Box::new($crate::registry::adapters::$out(
                            <$ty>::new(
                                $( $crate::ito_param_get!($pkind, values, ::core::stringify!($pname)), )*
                                ::stochastic_rs_stochastic::simd_rng::Unseeded,
                            ),
                        ))
                    }
                    build
                },
            }
        }
    };
}
