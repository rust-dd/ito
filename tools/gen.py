import os, re, sys

SRC = "/Users/danixx/Desktop/stochastic-rs/stochastic-rs-stochastic/src"
CATS = {
    "diffusion": "Diffusion",
    "volatility": "Volatility",
    "jump": "Jump",
    "interest": "Interest",
    "autoregressive": "Autoregressive",
    "rough": "Rough",
    "correlation": "Correlation",
}

def balanced(text, start):
    # start points at '('; return substring inside the matching parens
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return text[start+1:i]
        i += 1
    return ""

def split_top(s):
    out, depth, cur = [], 0, ""
    for c in s:
        if c in "(<[":
            depth += 1
        elif c in ")>]":
            depth -= 1
        if c == ',' and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += c
    if cur.strip():
        out.append(cur)
    return [x.strip() for x in out if x.strip()]

def norm(t):
    return re.sub(r"\s+", "", t)

def capture_type(text, eq_pos):
    # read from just after '=' until a ';' at bracket depth 0
    depth = 0
    i = eq_pos + 1
    out = ""
    while i < len(text):
        c = text[i]
        if c in "[(<{":
            depth += 1
        elif c in "])>}":
            depth -= 1
        elif c == ';' and depth == 0:
            return out
        out += c
        i += 1
    return out

def classify_param(ty):
    t = norm(ty)
    if t == "T": return "f64"
    if t == "usize": return "usize"
    if t == "Option<T>": return "opt_f64"
    if t == "Option<usize>": return "opt_usize"
    if t == "Option<bool>": return "opt_bool"
    if t == "bool": return "bool"
    if t == "Array1<T>": return "f64vec"
    if t == "Option<Array1<T>>": return "opt_f64vec"
    return None

def classify_output(out):
    t = norm(out)
    if t == "Array1<T>": return ("Path1D", None)
    m = re.match(r"\[Array1<T>;(\d+)\]", t)
    if m: return ("MultiDim", int(m.group(1)))
    return (None, None)

def extract(path):
    text = open(path).read()
    mimpl = re.search(r"impl[^\n]*ProcessExt<[^\n]*for\s+(\w+)", text)
    if not mimpl: return None
    struct = mimpl.group(1)
    meq = re.search(r"type\s+Output\s*=", text)
    if not meq: return None
    out_type = capture_type(text, text.index("=", meq.start()))
    okind, n = classify_output(out_type)
    mnew = re.search(r"pub fn new\s*\(", text)
    if not mnew: return None
    paren_start = text.index("(", mnew.start())
    args = split_top(balanced(text, paren_start))
    params = []
    ok = True
    reason = ""
    for a in args:
        if ':' not in a:
            continue
        name, ty = a.split(':', 1)
        name = name.strip()
        if name == "seed":
            continue
        kind = classify_param(ty)
        if kind is None:
            ok = False
            reason = f"{name}:{norm(ty)}"
            break
        params.append((name, kind))
    return dict(struct=struct, okind=okind, n=n, params=params, ok=ok, reason=reason)

def report():
    total = supported = 0
    skip_out = skip_param = 0
    for cat in CATS:
        d = os.path.join(SRC, cat)
        if not os.path.isdir(d):
            print(f"!! missing dir {d}"); continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".rs") and f != "mod.rs")
        print(f"\n=== {cat} ({len(files)} files) ===")
        for f in files:
            info = extract(os.path.join(d, f))
            total += 1
            mod = f[:-3]
            if info is None:
                print(f"  SKIP   {mod:24} (no ProcessExt/new)")
                continue
            if info["okind"] is None:
                skip_out += 1
                print(f"  skipΩ  {mod:24} {info['struct']:20} output not 1D/NxD")
                continue
            if not info["ok"]:
                skip_param += 1
                print(f"  skipΠ  {mod:24} {info['struct']:20} non-scalar param {info['reason']}")
                continue
            supported += 1
            tag = info["okind"] + (f"[{info['n']}]" if info['n'] else "")
            print(f"  OK     {mod:24} {info['struct']:20} {tag:10} {len(info['params'])}p")
    print(f"\nTOTAL files={total} supported={supported} skip_output={skip_out} skip_param={skip_param}")

OUT = "/Users/danixx/Desktop/ito/src/processes"

F64_DEFAULTS = {
    "hurst": "0.4", "rho": "-0.5", "rho1": "-0.5", "rho2": "-0.5", "rho3": "-0.5",
    "sigma": "0.2", "sigma1": "0.2", "sigma2": "0.2", "sigma3": "0.2", "sigmav": "0.2",
    "vol": "0.2", "eta": "0.4", "eta1": "3.0", "eta2": "3.0",
    "xi": "0.04", "nu": "0.2", "omega": "0.2", "epsilon": "0.3",
    "mu": "0.1", "kappa": "1.5", "theta": "0.5", "lambda": "1.0",
    "alpha": "0.5", "beta": "0.5", "gamma": "0.5", "delta": "0.5",
    "a": "0.5", "b": "0.5", "c": "1.0", "k": "1.0", "m": "5.0", "g": "5.0", "y": "0.5", "r": "0.03",
}
OPT_F64_DEFAULTS = {
    "t": "Some(1.0)", "x0": "Some(0.5)", "x": "Some(0.5)", "s0": "Some(100.0)",
    "v0": "Some(0.04)", "y0": "Some(0.5)", "x1_0": "Some(0.5)", "x2_0": "Some(0.5)",
    "r": "Some(0.03)", "r_f": "Some(0.02)", "mu": "Some(0.1)", "b": "Some(0.1)",
}
OVERRIDES = {
    ("Gbm", "mu"): "0.05", ("Gbm", "x0"): "Some(100.0)",
    ("Fgbm", "mu"): "0.05", ("Fgbm", "x0"): "Some(100.0)",
    ("Ou", "theta"): "1.0", ("Ou", "mu"): "1.2", ("Ou", "sigma"): "0.3",
    ("Cir", "theta"): "2.0", ("Cir", "mu"): "0.5",
    ("Fou", "theta"): "1.0", ("Fou", "mu"): "1.0", ("Fou", "sigma"): "0.3",
    ("Cev", "mu"): "0.05", ("Cev", "sigma"): "0.3", ("Cev", "gamma"): "0.8", ("Cev", "x0"): "Some(100.0)",
    ("Jacobi", "alpha"): "0.3", ("Jacobi", "beta"): "0.7",
    ("FJacobi", "alpha"): "0.3", ("FJacobi", "beta"): "0.7",
    ("HawkesJD", "alpha"): "0.5", ("HawkesJD", "beta"): "1.5",
}
DOCS = {
    "n": "Steps", "t": "Horizon", "hurst": "Hurst exponent", "mu": "Drift / mean",
    "sigma": "Diffusion scale", "theta": "Mean / reversion", "kappa": "Reversion speed",
    "rho": "Correlation", "lambda": "Jump intensity", "x0": "Initial value",
    "s0": "Initial spot", "v0": "Initial variance", "use_sym": "Symmetrise",
    "clamp": "Clamp positive", "degree": "Approximation degree",
}

def default_for(struct, name, kind):
    if (struct, name) in OVERRIDES:
        return OVERRIDES[(struct, name)]
    nm = name.lower()
    if kind == "f64": return F64_DEFAULTS.get(nm, "0.5")
    if kind == "usize":
        if nm == "n": return "1000"
        if nm == "j": return "50"
        if nm == "s": return "12"
        return "1"
    if kind == "opt_f64": return OPT_F64_DEFAULTS.get(nm, "Some(0.5)")
    if kind == "opt_usize": return "Some(1)"
    if kind == "opt_bool": return "Some(true)"
    if kind == "f64vec":
        return VEC_DEFAULTS.get(nm, "&[0.5, 0.3]")
    if kind == "opt_f64vec":
        return "None"
    return "0.5"

VEC_DEFAULTS = {
    "alpha": "&[0.1]", "beta": "&[0.85]", "gamma": "&[0.05]", "delta": "&[0.05]",
    "phi": "&[0.5]", "theta": "&[0.4]",
}

def doc_for(name):
    return DOCS.get(name.lower(), name)

def components_for(cat, okind, n):
    if okind != "MultiDim":
        return "[]"
    n = n or 2
    if cat in ("volatility", "rough"):
        names = ["asset", "variance", "variance 2", "variance 3"][:n]
    else:
        names = [f"x{k + 1}" for k in range(n)]
    while len(names) < n:
        names.append(f"comp {len(names) + 1}")
    return "[" + ", ".join(f'"{x}"' for x in names) + "]"

def emit():
    mods, grand = [], 0
    for cat, Cat in CATS.items():
        d = os.path.join(SRC, cat)
        files = sorted(f for f in os.listdir(d) if f.endswith(".rs") and f != "mod.rs")
        blocks, uses = [], []
        for f in files:
            info = extract(os.path.join(d, f))
            if not info or info["okind"] is None or not info["ok"]:
                continue
            mod, struct, params = f[:-3], info["struct"], info["params"]
            uses.append(f"use stochastic_rs_stochastic::{cat}::{mod}::{struct};")
            namew = max((len(n) for n, _ in params), default=1)
            kindw = max((len(k) for _, k in params), default=1)
            plines = [f'        {pn:<{namew}} : {pk:<{kindw}} = {default_for(struct, pn, pk)} ; "{doc_for(pn)}",'
                      for pn, pk in params]
            blocks.append(
                "process! {\n"
                f'    name: "{struct}",\n'
                f"    ty: {struct}<f64>,\n"
                f"    category: {Cat},\n"
                f"    output: {info['okind']},\n"
                f"    components: {components_for(cat, info['okind'], info['n'])},\n"
                "    params: [\n" + "\n".join(plines) + "\n    ],\n"
                "}"
            )
        if not blocks:
            continue
        mods.append(cat)
        grand += len(blocks)
        header = (f"//! {Cat} process registrations (generated by tools/gen.py).\n\n"
                  "use crate::process;\n" + "\n".join(sorted(set(uses))) + "\n\n")
        open(os.path.join(OUT, f"{cat}.rs"), "w").write(header + "\n\n".join(blocks) + "\n")
        print(f"wrote {cat}.rs: {len(blocks)} processes")
    modrs = ("//! Process registrations grouped by category. Each submodule populates the\n"
             "//! global registry through the `process!` macro.\n\n"
             + "\n".join(f"pub mod {m};" for m in sorted(mods)) + "\n")
    open(os.path.join(OUT, "mod.rs"), "w").write(modrs)
    print(f"wrote mod.rs: {sorted(mods)}  total={grand}")

emit()
