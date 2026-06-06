# ito

A terminal UI for exploring the [`stochastic-rs`](https://github.com/rust-dd/stochastic-rs)
process library. Pick any process by name, edit every parameter, generate `M`
Monte-Carlo paths, and plot them — all on the CPU, in `f64`.

```
cargo run
```

## Keys

| Key | Action |
| --- | --- |
| `↑` / `↓` (or `j` / `k` in the list) | move selection / field |
| `Tab` | switch between the process list and the parameter form |
| `Enter` / `g` | generate paths |
| `1`–`9` (in the list) | switch the chart between path types — each state variable (asset / variance / …) on its own scale. The title shows the bound keys as `[1][2][3]…` and names the selected one |
| `v` (in the list) | toggle a grid of every path-type chart vs. the single paged view |
| `/` | filter the process list (Enter to apply, Esc to clear) |
| type / `Backspace` | edit the focused parameter (in the form) |
| `q` / `Ctrl-C` | quit |

Each parameter shows its type hint: `f64`, `uint`, `f64?`/`uint?`/`bool?`
(optional — type `none` to leave unset). `paths (M)` controls how many
independent trajectories are drawn.

## How it works

`stochastic-rs` is consumed as a git dependency and is **not** modified. Because
Rust has no reflection, every process is described explicitly through the
`process!` macro (`src/registry/macros.rs`), which mirrors the real `new(...)`
constructor and registers a [`ProcessDescriptor`](src/registry/mod.rs) via
`inventory`. Output adapters (`src/registry/adapters.rs`) normalise the
heterogeneous process outputs (`Array1`, `[Array1; N]`, `Array2`, `Complex`,
…) into plottable series.

## Coverage

**All 118 `ProcessExt` processes** in `stochastic-rs` are registered and sample
cleanly with their default parameters — across diffusion, volatility, jump,
interest, rough, correlation, autoregressive, noise, process, and sheet
categories.

The scalar- and vector-parameter processes use the `process!` macro
(`src/processes/*.rs`). The rest are hand-written in `src/processes/manual.rs`,
each pinning the non-scalar constructor argument to a sensible value:

- **fixed enum** — Heston (`Sqrt` scheme), transformed-OU (`Tanh`).
- **jump distributions** — Merton, Lévy, Bates, jump-FOU (Normal), Kou
  (double-exponential), custom-jump FOU / CustomJt / Ctrw / CompoundPoisson.
- **callable term-structures** (flat curve) — Hull-White, Hull-White-2F,
  Ho-Lee, ADG, HJM.
- **fixed-size arrays / matrices** — 2-asset & multi-factor Heston, multi-factor
  SABR, regime-switching (transition matrix), multivariate Hawkes.
- **nested processes** — 2-factor CIR.
- **non-`Array1` outputs** — `Array2` curves, `Vec<Array1>`, `Complex` (cfou),
  and the spot-plus-variances tuple — handled by dedicated chart adapters.

See [`docs/superpowers/specs/2026-06-05-ito-design.md`](docs/superpowers/specs/2026-06-05-ito-design.md)
for the design.
