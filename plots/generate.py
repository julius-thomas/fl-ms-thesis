from __future__ import annotations

import argparse
import yaml
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from os.path import isfile, join

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

try:
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "font.size": 14,
            "legend.fontsize": 14,
        }
    )
except Exception as e:
    print(e)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TB_DIR = os.path.join(THIS_DIR, "tb_objects")
PDF_DIR = os.path.join(THIS_DIR, "pdf")
CONFIG_PATH = os.path.join(THIS_DIR, "config.yaml")


# ---------------- Helpers ----------------

def _r4(x):
    return None if x is None else round(float(x), 4)


def _ratio_label(r):
    if r is None:
        return ""
    r = float(r)
    if abs(r - 0.98) < 1e-12:
        r = 0.99
    return f"{r:.2f}"


def _resolve_array(spec):
    """Accept a list, a linspace dict, or None."""
    if spec is None:
        return None
    if isinstance(spec, dict) and spec.get("type") == "linspace":
        return np.linspace(spec["start"], spec["end"], spec["num"])
    return np.asarray(spec)


def _write_recovery_csv(rows, csv_path):
    if not rows:
        return
    df = pd.DataFrame(
        rows,
        columns=[
            "plot_name",
            "algo_name",
            "target_threshold",
            "recovery_round",
            "final_accuracy",
        ],
    )
    df["recovery_round"] = pd.to_numeric(df["recovery_round"], errors="coerce").astype("Int64")
    m = df["plot_name"].astype(str).str.extract(r"^fig\s*(\d+)([a-zA-Z]?)", expand=True)
    df["_fig_num"] = pd.to_numeric(m[0], errors="coerce").fillna(10**9).astype(int)
    df["_fig_letter"] = m[1].fillna("").str.lower()
    df = df.sort_values(
        by=["_fig_num", "_fig_letter", "plot_name", "algo_name"],
        kind="mergesort",
    ).drop(columns=["_fig_num", "_fig_letter"])
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False, na_rep="", float_format="%.4f")


# ---------------- TB object generation ----------------

def convert_tb_data(root_dir, sort_by=None):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    rows = []
    for root, _, filenames in os.walk(root_dir):
        if not any("events.out.tfevents" in f for f in filenames):
            continue
        ea = EventAccumulator(root, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            for e in ea.Scalars(tag):
                rows.append((e.wall_time, tag, e.step, float(e.value)))
    if not rows:
        return pd.DataFrame(columns=["wall_time", "name", "step", "value"])
    df = pd.DataFrame(rows, columns=["wall_time", "name", "step", "value"])
    if sort_by is not None:
        df = df.sort_values(sort_by)
    return df.reset_index(drop=True)


def aggregate_dfs(dir_path, df_names, delimiter):
    if not df_names:
        return {}
    found = []
    for root_dir, subdirs, _ in os.walk(dir_path):
        for item in subdirs:
            key = item if delimiter is None else item.split(delimiter)[0]
            if key in df_names:
                found.append(os.path.join(root_dir, item))
    ordered = []
    for short in df_names:
        tag = short + (delimiter or "")
        for exp in found:
            if tag in exp:
                ordered.append(exp)
    dfs = {}
    for exp in ordered:
        base = os.path.basename(exp)
        key = base.split(delimiter)[0] if delimiter else base
        dfs[key] = convert_tb_data(exp, sort_by=["name", "step"])
    return dfs


def _eval_function(fn_str, length):
    xs = range(length) if isinstance(length, int) else length
    env = {"pow": pow, "math": math, "np": np}
    return pd.Series([eval(fn_str, {"__builtins__": {}}, {**env, "x": x}) for x in xs])


def accumulate_dfs(dfs, property_name, functions, function_length, push_front=False):
    out = pd.DataFrame()
    cols = []
    for k in dfs:
        t = (
            dfs[k]
            .loc[dfs[k]["name"] == property_name]
            .reset_index()
            .drop_duplicates(subset=["step"])
        )
        cols.append(t["value"])
    if cols:
        function_length = min(len(c.index) for c in cols)
    if push_front:
        for i, fn in enumerate(functions):
            out[f"function_{i}"] = _eval_function(fn, function_length)
    for i, key in enumerate(dfs):
        c = cols[i]
        n = len(c.index) - function_length
        out[key] = c.drop(c.tail(n).index) if n > 0 else c
    out.index = np.arange(1, len(out) + 1)
    return out


def build_tb_object(source_cfg, out_path):
    if "lines" in source_cfg:
        out = _build_from_lines(source_cfg)
    else:
        out = _build_from_experiments(source_cfg)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_pickle(out_path)


def _build_from_experiments(source_cfg):
    path = source_cfg["path"]
    exp_names = source_cfg.get("experiment_names", [])
    prop = source_cfg.get("property_name", "")
    delim = source_cfg.get("name_delimiter")
    fns = source_cfg.get("functions", [])
    fn_len = source_cfg.get("function_length")
    if isinstance(fn_len, dict) and fn_len.get("type") == "linspace":
        fn_len = np.linspace(fn_len["start"], fn_len["end"], fn_len["num"])
    dfs = aggregate_dfs(path, exp_names, delim)
    return accumulate_dfs(dfs, prop, fns, fn_len, push_front=bool(fns))


def _load_run_series(run_dir, tag):
    df = convert_tb_data(run_dir, sort_by=["step"])
    s = df.loc[df["name"] == tag].drop_duplicates(subset=["step"])
    return s["value"].reset_index(drop=True)


def _build_from_lines(source_cfg):
    """Average multiple runs per line.

    source = {
      "tag": "Server Acc1",
      "lines": [
        {"name": "FedAvg", "path": "~/logs/FEDAVG", "match": "PREFIX_seed"},
        ...
      ]
    }

    For each line, emits columns:
      <name>, <name>__std_lo/hi (mean ± sample std), <name>__iqr_lo/hi (25/75 pct).
    """
    tag = source_cfg["tag"]
    lines = source_cfg["lines"]
    out = pd.DataFrame()

    for line in lines:
        name = line["name"]
        base = os.path.expanduser(line["path"])
        match = line["match"]

        if not os.path.isdir(base):
            print(f"[warn] {name}: path does not exist: {base}")
            continue

        run_dirs = sorted(
            os.path.join(base, e)
            for e in os.listdir(base)
            if e.startswith(match) and os.path.isdir(os.path.join(base, e))
        )
        if not run_dirs:
            print(f"[warn] {name}: no runs at {base} matching {match!r}")
            continue

        runs = [_load_run_series(d, tag) for d in run_dirs]
        runs = [r for r in runs if len(r) > 0]
        if not runs:
            print(f"[warn] {name}: tag {tag!r} not found in any run")
            continue

        n = min(len(r) for r in runs)
        arr = np.stack([r.iloc[:n].to_numpy() for r in runs], axis=1)  # (n, k)

        mean = arr.mean(axis=1)
        std = arr.std(axis=1, ddof=1) if arr.shape[1] > 1 else np.zeros(arr.shape[0])
        q1 = np.percentile(arr, 25, axis=1)
        q3 = np.percentile(arr, 75, axis=1)

        if out.empty:
            out.index = np.arange(1, n + 1)
        elif n < len(out):
            out = out.iloc[:n].copy()
            out.index = np.arange(1, n + 1)
        elif n > len(out):
            m = len(out)
            mean, std, q1, q3 = mean[:m], std[:m], q1[:m], q3[:m]

        out[name] = mean
        out[f"{name}__std_lo"] = mean - std
        out[f"{name}__std_hi"] = mean + std
        out[f"{name}__iqr_lo"] = q1
        out[f"{name}__iqr_hi"] = q3

        print(f"[agg] {name}: {arr.shape[1]} runs × {arr.shape[0]} steps from {base}")

    return out


# ---------------- Plotting ----------------

def smooth_line(scalars, weight, first_it=True):
    last = scalars[0]
    smoothed = []
    for p in scalars:
        v = last * weight + (1 - weight) * p
        smoothed.append(v)
        last = v
    if first_it:
        ema2 = smooth_line(smoothed, weight, False)
        smoothed = list((2 * np.array(smoothed)) - np.array(ema2))
    return smoothed


def create_plot(df, cfg, output, plot_name_for_csv=None):
    plt.clf()
    fig, ax = plt.subplots()

    legend = cfg.get("legend", [])
    labels = cfg["labels"]
    title = cfg.get("title")
    ylim = cfg.get("ylim")
    xlim = cfg.get("xlim")
    scale = cfg.get("scale")
    smooth = cfg.get("smooth", False)
    weight = cfg.get("weight") if cfg.get("weight") is not None else 0.5
    x_values = _resolve_array(cfg.get("x_values"))
    iqr_flags = cfg.get("iqr_flags")
    recovery = cfg.get("recovery")

    x = df.index.to_numpy() if x_values is None else np.asarray(x_values)

    band_suffixes = ("__iqr_lo", "__iqr_hi", "__std_lo", "__std_hi")
    base_cols = [c for c in df.columns if not any(str(c).endswith(s) for s in band_suffixes)]

    def _norm_flags(flags):
        if flags is None:
            return [False] * len(base_cols)
        if isinstance(flags, bool):
            return [flags] * len(base_cols)
        return (list(flags) + [False] * len(base_cols))[: len(base_cols)]

    iqr_norm = _norm_flags(iqr_flags)
    std_norm = _norm_flags(cfg.get("std_flags"))

    if recovery is None:
        rec_enabled = [False] * len(base_cols)
        rec_x0 = rec_ratio = rec_x_start = None
        rec_mode = "per_curve"
        rec_marker_size = 70
        rec_print = False
    else:
        en = recovery.get("enabled", True)
        rec_enabled = (
            [en] * len(base_cols)
            if isinstance(en, bool)
            else (list(en) + [False] * len(base_cols))[: len(base_cols)]
        )
        rec_x0 = recovery.get("x0")
        rec_ratio = recovery.get("ratio", 0.99)
        rec_mode = recovery.get("mode", "per_curve")
        rec_marker_size = recovery.get("marker_size", 70)
        rec_print = bool(recovery.get("print", True))
        rec_x_start = recovery.get("x_start")

    def _idx(x_arr, x0):
        if x0 is None:
            return None
        m = np.where(x_arr == x0)[0]
        if len(m):
            return int(m[0])
        try:
            return int(np.argmin(np.abs(x_arr.astype(float) - float(x0))))
        except Exception:
            return None

    y_by = {
        c: np.asarray(
            smooth_line(df[c].to_numpy().tolist(), weight) if smooth else df[c].to_numpy()
        )
        for c in base_cols
    }

    idx0 = _idx(x, rec_x0) if rec_x0 is not None else None
    if rec_x_start is None and rec_x0 is not None:
        rec_x_start = rec_x0 + 1
    idx_start = _idx(x, rec_x_start) if rec_x_start is not None else None

    global_baseline = None
    if rec_x0 is not None and idx0 is not None and rec_mode == "global_max_at_x0":
        vals = [y_by[c][idx0] for c in base_cols if 0 <= idx0 < len(y_by[c])]
        global_baseline = max(vals) if vals else None

    if rec_print and rec_x0 is not None and any(rec_enabled):
        print(
            f"\n[recovery] plot={plot_name_for_csv or output}  x0={rec_x0}  "
            f"x_start={rec_x_start}  ratio={rec_ratio}  mode={rec_mode}"
        )
        if rec_mode == "global_max_at_x0":
            print(f"[recovery] global_baseline_at_x0={global_baseline}")

    recovery_rows = []
    for i, col in enumerate(base_cols):
        y = y_by[col]
        if legend:
            (line,) = ax.plot(x, y, label=legend[i])
            lab = legend[i]
        else:
            (line,) = ax.plot(x, y)
            lab = str(col)
        color = line.get_color()

        if iqr_norm[i]:
            lo_col, hi_col = f"{col}__iqr_lo", f"{col}__iqr_hi"
            if lo_col in df.columns and hi_col in df.columns:
                y_lo, y_hi = df[lo_col].to_numpy(), df[hi_col].to_numpy()
                ax.plot(x, y_lo, color=color, alpha=0.5, linewidth=1)
                ax.plot(x, y_hi, color=color, alpha=0.5, linewidth=1)
                ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.2)

        if std_norm[i]:
            lo_col, hi_col = f"{col}__std_lo", f"{col}__std_hi"
            if lo_col in df.columns and hi_col in df.columns:
                y_lo, y_hi = df[lo_col].to_numpy(), df[hi_col].to_numpy()
                ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.2)

        if rec_enabled[i] and rec_x0 is not None and rec_ratio is not None:
            pname = plot_name_for_csv or (title or output)
            if idx0 is None or not (0 <= idx0 < len(y)):
                if rec_print:
                    print(f"  - {lab}: x0 not found / out of range")
                recovery_rows.append({
                    "plot_name": pname, "algo_name": lab,
                    "target_threshold": _ratio_label(rec_ratio),
                    "recovery_round": None,
                    "final_accuracy": _r4(y[-1]) if len(y) else None,
                })
                continue
            baseline = global_baseline if rec_mode == "global_max_at_x0" else y[idx0]
            if baseline is None or np.isnan(baseline):
                if rec_print:
                    print(f"  - {lab}: baseline is None/NaN")
                recovery_rows.append({
                    "plot_name": pname, "algo_name": lab,
                    "target_threshold": _ratio_label(rec_ratio),
                    "recovery_round": None,
                    "final_accuracy": _r4(y[-1]) if len(y) else None,
                })
                continue
            target = rec_ratio * baseline
            rec_idx = None
            start_j = (idx0 + 1) if idx_start is None else max(idx_start, idx0 + 1)
            for j in range(start_j, len(y)):
                if y[j] >= target:
                    rec_idx = j
                    break
            rec_round = None
            if rec_idx is not None:
                rec_round = int(round(float(x[rec_idx])))
                ax.scatter(
                    [x[rec_idx]], [y[rec_idx]],
                    marker="x", s=rec_marker_size,
                    color=color, linewidths=2, zorder=5,
                )
                if rec_print:
                    print(f"  - {lab}: rec_at_round={rec_round}  target={_r4(target):.4f}")
            else:
                if rec_print:
                    print(f"  - {lab}: not reached  target={_r4(target):.4f}")
            recovery_rows.append({
                "plot_name": pname, "algo_name": lab,
                "target_threshold": _ratio_label(rec_ratio),
                "recovery_round": rec_round,
                "final_accuracy": _r4(y[-1]) if len(y) else None,
            })

    if scale == "exp":
        ax.set_yscale("function", functions=(partial(np.power, 10.0), np.log10))

    if title:
        ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)

    hlines = cfg.get("hlines")
    if hlines is not None:
        items = hlines if isinstance(hlines, (list, tuple)) else [hlines]
        for h in items:
            if isinstance(h, dict):
                y = h["y"]
                color = h.get("color", "0.4")
                lw = h.get("linewidth", 0.8)
                ls = h.get("linestyle", "--")
                label = h.get("label")
            else:
                y, color, lw, ls, label = h, "0.4", 0.8, "--", None
            ax.axhline(y=y, color=color, linewidth=lw, linestyle=ls, label=label)

    vlines = cfg.get("vlines")
    if vlines is not None:
        items = vlines if isinstance(vlines, (list, tuple)) else [vlines]
        for v in items:
            if isinstance(v, dict):
                xv = v["x"]
                color = v.get("color", "0.4")
                lw = v.get("linewidth", 0.8)
                ls = v.get("linestyle", "--")
                label = v.get("label")
            else:
                xv, color, lw, ls, label = v, "0.4", 0.8, "--", None
            ax.axvline(x=xv, color=color, linewidth=lw, linestyle=ls, label=label)

    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(labels["x"])
    ax.set_ylabel(labels["y"])
    if legend:
        ax.legend()

    plt.savefig(output, dpi=1000, bbox_inches="tight")
    plt.close(fig)
    return recovery_rows


# ---------------- Orchestration ----------------

def generate_objects(config, tb_dir, workers=1):
    os.makedirs(tb_dir, exist_ok=True)
    seen = set()
    jobs = []
    for name, entry in config.get("plots", {}).items():
        df_name = entry.get("df_name", name)
        if df_name in seen:
            continue
        seen.add(df_name)
        out_path = os.path.join(tb_dir, df_name)
        if isfile(out_path):
            print(f"[skip] object exists: {df_name}")
            continue
        src = entry.get("source")
        if not src:
            print(f"[miss] no source for {df_name} — place pickle in {tb_dir}")
            continue
        jobs.append((df_name, src, out_path))

    if not jobs:
        return

    if workers <= 1 or len(jobs) == 1:
        for df_name, src, out_path in jobs:
            print(f"[gen]  {df_name}")
            build_tb_object(src, out_path)
        return

    n = min(workers, len(jobs))
    print(f"[gen]  dispatching {len(jobs)} objects across {n} threads")
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = {ex.submit(build_tb_object, src, out_path): df_name
                for df_name, src, out_path in jobs}
        for fut in as_completed(futs):
            df_name = futs[fut]
            try:
                fut.result()
                print(f"[done] {df_name}")
            except Exception as e:
                print(f"[fail] {df_name}: {type(e).__name__}: {e}")


def generate_plots(config, tb_dir, pdf_dir, csv_path=None):
    os.makedirs(pdf_dir, exist_ok=True)
    all_rows = []
    for name, entry in config.get("plots", {}).items():
        df_name = entry.get("df_name", name)
        pkl = os.path.join(tb_dir, df_name)
        if not isfile(pkl):
            print(f"[skip] {name}: object not found at {pkl}")
            continue
        try:
            df = pd.read_pickle(pkl)
        except Exception as e:
            print(f"[skip] {name}: cannot read pickle ({type(e).__name__}: {e})")
            continue
        plot_cfg = entry["plot"]
        out = os.path.join(pdf_dir, entry.get("output_name") or name)
        rows = create_plot(
            df, plot_cfg, out,
            plot_name_for_csv=plot_cfg.get("title") or name,
        )
        all_rows.extend(rows)
    if csv_path:
        _write_recovery_csv(all_rows, csv_path)
        print(f"[recovery] wrote csv: {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--tb-dir", default=TB_DIR)
    ap.add_argument("--pdf-dir", default=PDF_DIR)
    ap.add_argument("--csv", default=None, help="recovery-summary csv path")
    ap.add_argument("--skip-objects", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 1)),
                    help="parallel threads for object generation (default: min(8, cpu_count))")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if not args.skip_objects:
        generate_objects(config, args.tb_dir, workers=args.workers)
    if not args.skip_plots:
        csv = args.csv or os.path.join(args.pdf_dir, "recovery_summary.csv")
        generate_plots(config, args.tb_dir, args.pdf_dir, csv_path=csv)


if __name__ == "__main__":
    main()
