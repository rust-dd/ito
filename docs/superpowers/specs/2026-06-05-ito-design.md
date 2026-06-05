# ito — design

A terminal UI (ratatui) to explore the `stochastic-rs` process library: pick any
process by name, edit every parameter, generate `M` Monte-Carlo paths, and plot
them on a chart. CPU backend, `f64`, single binary crate.

## Decisions (locked)

- **Name:** `ito` (after Itô calculus).
- **stochastic-rs:** consumed as a **git dependency** (`rust-dd/stochastic-rs`,
  `main`, v2.3) — **not modified**. All registry/adapter/macro code lives in `ito`.
- **Backend:** CPU only. Type fixed to `f64`. Seed strategy `Unseeded`, so the
  `M` paths of `sample_par(M)` are genuinely independent Monte-Carlo draws.
- **ndarray:** pinned to `0.17` to unify with the library's `Array1<f64>` output.
- **Generation semantics:** select **one** configured process, choose `M`,
  call `sample_par(M)`, overlay the `M` trajectories.

## Why a hand-written registry

Rust has no runtime reflection, and `stochastic-rs` exposes ~120-140 processes as
structs with **positional** `new(...)` constructors whose argument lists do **not**
match struct field order (constructors take args like `x0`/`s0`/`v0` that aren't
stored as fields, reorder them, and compute composite fields). So a uniform
"list every process + expose every parameter" layer must be authored explicitly.

We keep per-process cost low with a local `process!` `macro_rules!` macro that
mirrors the real constructor signature and generates: the `ParamSpec` list, a
type-erased `build` closure that calls the real `new(...)`, the output adapter,
and an `inventory` registration. Because it compiles against the real `new`, any
mismatch is a compile error.

## Architecture (single crate)

```
src/
  main.rs            terminal setup + event loop
  registry/
    mod.rs           ParamKind, ParamSpec, ParamValues, Category,
                     NamedSeries, ChartSource, ProcessDescriptor, registry()
    adapters.rs      Path1D / MultiDim<N> / Curve adapters → ChartSource
    macros.rs        process! macro
  processes/
    diffusion.rs     process!{ Gbm<f64>, ... } registrations (one file per
    volatility.rs    category, kept under the 600-line cap)
    jump.rs
    interest.rs
    ...
  app/
    state.rs         App: selected process, param form, M, generated series
    event.rs         key handling
    ui.rs            ratatui rendering: list | form | chart
```

### Registry types

- `ParamKind ∈ {F64, Usize, OptF64, Bool}` — drives editing + coercion.
- `ParamSpec { name, kind, default, doc }` — static, per constructor argument.
- `ParamValues` — runtime name→value map with typed getters (`f64`, `usize`,
  `opt_f64`, `bool`), seeded from the specs' defaults.
- `ChartSource: Send + Sync` — `fn sample_par(&self, m) -> Vec<Vec<NamedSeries>>`
  (m samples, each ≥1 labelled series: 1 for 1-D, 2 for Heston/SABR, …).
- `ProcessDescriptor { name, category, params, build }` collected via
  `inventory`; `registry()` returns all, sorted by `(category, name)`.

### Output adapters

`Path1D<P>` wraps `P: ProcessExt<f64, Output = Array1<f64>>`; `MultiDim<P, N>`
wraps `[Array1<f64>; N]`. Each converts a sample into labelled series with the
time axis on `x` (`i·dt`, `dt = t/(n-1)`, else index).

## TUI

```
┌ Processes ─────┐┌ Parameters: Ou ───────────┐
│ ▸ Diffusion    ││ theta [ 1.0 ]  mu [ 1.2 ] │
│   Gbm  Ou ◀ Cir││ sigma [ 0.3 ]  n  [ 1000 ]│
│ ▸ Volatility   ││ x0 [0.5] t [(none)]       │
│   Heston Sabr  ││ paths M [ 10 ]   [g] gen  │
└────────────────┘└───────────────────────────┘
┌ Chart: 10 paths of Ou ─────────────────────┐
│            (ratatui Chart, M overlay)       │
└─────────────────────────────────────────────┘
 Tab pane · ↑↓ nav · ⏎/g generate · / filter · q quit
```

## Rollout

Incremental, with `feat:` commits (no scope, no co-author): macro + registry +
adapters → diffusion category end-to-end with a working TUI → then category by
category. The registry auto-collects whatever is annotated. Exotic tail (custom
jump distributions, curve/sheet, variable-dim Hawkes) gets a manual `build`
behind the same `ProcessDescriptor` interface, or is deferred.

## Out of scope (v1)

GPU/CUDA backend, `f32`, reproducible seeding, export/save, parameter
range validation beyond type coercion.
