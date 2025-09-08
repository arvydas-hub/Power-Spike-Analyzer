#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Power Spike Analyzer — GUI + CLI (single-file edition)

This file combines:
  • The PowerSentry SD analyzer (voltage/frequency parsing, metrics, plots, reports)
  • A Tkinter GUI to pick an SD dump folder and run analyses with buttons

Dependencies (install into your venv):
  pip install pandas matplotlib

Run:
  • GUI (default when no args):  python main.py
  • CLI (same flags as before):  python main.py --dir <sd_dir> --out <report_dir> [--filter-freq] [--hist]
                                  or: python main.py --level <BASE>.csv --freq <BASE>F.csv --out <report_dir>

Notes:
  • The GUI writes all outputs to the chosen output directory.
  • "Analyze Selected" runs only the highlighted capture in the list.
  • "Analyze All" processes everything found and writes an index.md with links.
  • "Open Output Folder" opens the output directory in your OS file browser.
  • "View Last Report" opens the most recently generated report.
  • "Quit" exits (obviously).
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

# Core deps
import pandas as pd

matplotlib.use("Agg")  # headless-safe for plotting to files
# GUI (stdlib)
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import matplotlib.pyplot as plt

# ---------------------------
# Configurable constants
# ---------------------------
V_OK_MIN = 114.0
V_OK_MAX = 126.0
V_SPIKE = 135.0

F_CENTER = 60.0
F_TIGHT = 0.05
F_LOOSE = 0.20
F_HARD_MIN = 58.0
F_HARD_MAX = 62.0


@dataclass(frozen=True)
class SeriesFiles:
    """Holds discovered file paths for a base capture id."""

    base: str
    level_path: Optional[Path]
    freq_path: Optional[Path]


def _norm_cols(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        c2 = c.strip().lower()
        c2 = re.sub(r"\s+", "_", c2)
        out.append(c2)
    return out


def parse_base_from_filename(p: Path) -> Optional[str]:
    """Extract 12-digit base from filenames like 090725051139.csv or 090725051139F.csv."""
    m = re.search(r"(\d{12})(?:F)?\.csv$", p.name, flags=re.IGNORECASE)
    return m.group(1) if m else None


def parse_base_ts(base: str) -> Optional[datetime]:
    """Parse MMDDYYhhmmss to datetime (local naive)."""
    try:
        return datetime.strptime(base, "%m%d%y%H%M%S")
    except Exception:
        return None


def load_capture_csv(path: Path, value_guess: Optional[str] = None) -> pd.DataFrame:
    """
    Reads a CSV, ignoring '#' comment lines.
    Normalizes column names.
    Determines 'value' column automatically if not given.
    Builds 'ts' column from time_local or filename base + ms.

    Returns a tidy DataFrame with columns ['ts', 'value'], sorted by ts, NaNs dropped.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, comment="#", dtype=str).copy()
    if df.shape[0] == 0 and df.shape[1] == 0:
        return pd.DataFrame(columns=["ts", "value"])

    df.columns = _norm_cols(list(df.columns))

    # Identify value column
    cols = set(df.columns)
    candidates_ordered = []
    if value_guess:
        candidates_ordered.append(value_guess.strip().lower())
    candidates_ordered += [
        "level_v",
        "level",
        "voltage",
        "volts",
        "freq_hz",
        "frequency",
        "freq",
    ]

    value_col = None
    for c in candidates_ordered:
        if c in cols:
            value_col = c
            break
    if value_col is None:
        for c in df.columns:
            try:
                pd.to_numeric(df[c])
                value_col = c
                break
            except Exception:
                continue

    if value_col is None:
        return pd.DataFrame(columns=["ts", "value"])

    df["value"] = pd.to_numeric(df[value_col], errors="coerce")

    base = parse_base_from_filename(path) or ""
    base_ts = parse_base_ts(base)

    ts = None
    if "time_local" in df.columns:
        ts = pd.to_datetime(
            df["time_local"], errors="coerce", infer_datetime_format=True
        )
        if ts.isna().mean() > 0.5:
            fmts = ["%m/%d/%y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%m-%d-%y %H:%M:%S"]
            ts_try = None
            for fmt in fmts:
                try:
                    ts_try = pd.to_datetime(
                        df["time_local"], format=fmt, errors="coerce"
                    )
                except Exception:
                    ts_try = None
                if ts_try is not None and ts_try.notna().any():
                    ts = ts_try
                    break
    if ts is None or ts.isna().all():
        if base_ts is not None:
            ts = pd.Series([base_ts] * len(df), index=df.index, dtype="datetime64[ns]")
        else:
            ts = pd.to_datetime(pd.Series([pd.NaT] * len(df)), errors="coerce")

    if "ms" in df.columns:
        ms = pd.to_numeric(df["ms"], errors="coerce").fillna(0).astype("Int64")
        ts = ts + pd.to_timedelta(ms.fillna(0).astype("float") / 1000.0, unit="s")

    out = pd.DataFrame({"ts": ts, "value": df["value"]})
    out = out.dropna(subset=["ts", "value"]).sort_values("ts").reset_index(drop=True)
    return out


def compute_basic_stats(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "count": 0,
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "median": math.nan,
            "std_pop": math.nan,
            "rms": math.nan,
            "start_ts": None,
            "end_ts": None,
            "duration_s": math.nan,
        }

    v = df["value"].astype(float)
    count = int(v.count())
    v2 = (v**2).mean()
    rms = math.sqrt(v2) if pd.notna(v2) else math.nan
    start = df["ts"].iloc[0]
    end = df["ts"].iloc[-1]
    dur = (
        (end - start).total_seconds() if pd.notna(end) and pd.notna(start) else math.nan
    )

    return {
        "count": count,
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "median": float(v.median()),
        "std_pop": float(v.std(ddof=0)),
        "rms": float(rms),
        "start_ts": start,
        "end_ts": end,
        "duration_s": float(dur),
    }


def voltage_metrics(
    df: pd.DataFrame, v_ok_min: float, v_ok_max: float, v_spike: float
) -> Dict[str, float]:
    if df.empty:
        return {
            "pct_within_ok_band": math.nan,
            "spike_count_over_135v": 0,
        }
    v = df["value"].astype(float)
    within = ((v >= v_ok_min) & (v <= v_ok_max)).mean() * 100.0
    spike_count = int((v > v_spike).sum())
    return {
        "pct_within_ok_band": float(within),
        "spike_count_over_135v": spike_count,
    }


def frequency_metrics(
    df: pd.DataFrame,
    f_center: float,
    tight: float,
    loose: float,
    f_hard_min: float,
    f_hard_max: float,
) -> Dict[str, float]:
    if df.empty:
        return {
            "pct_within_tight": math.nan,
            "pct_within_loose": math.nan,
            "count_below_58hz": 0,
            "count_above_62hz": 0,
        }
    v = df["value"].astype(float)
    v = v[v > 0]
    if v.empty:
        return {
            "pct_within_tight": math.nan,
            "pct_within_loose": math.nan,
            "count_below_58hz": 0,
            "count_above_62hz": 0,
        }
    within_tight = (
        (v >= (f_center - tight)) & (v <= (f_center + tight))
    ).mean() * 100.0
    within_loose = (
        (v >= (f_center - loose)) & (v <= (f_center + loose))
    ).mean() * 100.0
    below = int((v < f_hard_min).sum())
    above = int((v > f_hard_max).sum())
    return {
        "pct_within_tight": float(within_tight),
        "pct_within_loose": float(within_loose),
        "count_below_58hz": below,
        "count_above_62hz": above,
    }


def apply_frequency_isolated_filter(
    df: pd.DataFrame, f_hard_min: float, f_hard_max: float
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Drop only isolated out-of-band singletons: keep clusters where a neighbor is also out-of-band.
    Returns filtered df and counts dict.
    """
    if df.empty:
        return df.copy(), {"total_out_of_band": 0, "dropped_isolated_singletons": 0}

    v = df["value"].astype(float)
    oob = (v < f_hard_min) | (v > f_hard_max)
    prev_oob = oob.shift(1, fill_value=False)
    next_oob = oob.shift(-1, fill_value=False)
    isolated = oob & (~prev_oob) & (~next_oob)

    filtered = df.loc[~isolated].reset_index(drop=True)
    counts = {
        "total_out_of_band": int(oob.sum()),
        "dropped_isolated_singletons": int(isolated.sum()),
    }
    return filtered, counts


def parse_device_footers(path: Path) -> Dict[str, str]:
    """Scan trailing comment lines '# key=value' and return a dict."""
    out: Dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[::-1]:
            if not line.strip().startswith("#"):
                break
            txt = line.strip()[1:].strip()
            m = re.match(r"([^=]+)=(.*)$", txt)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()
                out[key] = val
    except Exception:
        pass
    return out


def plot_series(df: pd.DataFrame, title: str, ylabel: str, out_png_path: Path) -> None:
    if df.empty:
        plt.figure()
        plt.title(title + " (no data)")
        plt.xlabel("Time")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out_png_path)
        plt.close()
        return

    plt.figure()
    plt.plot(df["ts"], df["value"])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png_path)
    plt.close()


def plot_hist(
    df: pd.DataFrame, title: str, xlabel: str, out_png_path: Path, bins: int = 50
) -> None:
    if df.empty:
        plt.figure()
        plt.title(title + " (no data)")
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_png_path)
        plt.close()
        return

    plt.figure()
    plt.hist(df["value"].astype(float).dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png_path)
    plt.close()


def _fmt_dt(dt: Optional[datetime]) -> str:
    if dt is None or (isinstance(dt, float) and math.isnan(dt)):
        return ""
    try:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(dt)


def _dict_to_markdown_table(d: Dict[str, object]) -> str:
    keys = list(d.keys())
    preferred = [
        "count",
        "min",
        "max",
        "mean",
        "median",
        "std_pop",
        "rms",
        "start_ts",
        "end_ts",
        "duration_s",
    ]
    keys_sorted = preferred + [k for k in keys if k not in preferred]
    seen = set()
    keys_sorted = [k for k in keys_sorted if not (k in seen or seen.add(k))]

    lines = ["| metric | value |", "|---|---|"]
    for k in keys_sorted:
        v = d.get(k, "")
        if isinstance(v, float) and math.isnan(v):
            s = ""
        elif isinstance(v, float):
            s = f"{v:.6g}"
        elif isinstance(v, datetime):
            s = _fmt_dt(v)
        else:
            s = str(v)
        lines.append(f"| {k} | {s} |")
    return "\n".join(lines)


def render_report(
    base: str,
    paths: Dict[str, Optional[Path]],
    stats_level: Optional[Dict[str, object]],
    stats_freq: Optional[Dict[str, object]],
    extras: Dict[str, Dict[str, object]],
    figures: Dict[str, Path],
    filtered_extras: Optional[Dict[str, object]] = None,
    footers: Optional[Dict[str, Dict[str, str]]] = None,
    out_md_path: Optional[Path] = None,
) -> str:
    parts: List[str] = []
    parts.append(f"# PowerSentry Capture Report — {base}\n")
    parts.append("## Files\n")
    parts.append(f"- Level CSV: `{paths.get('level')}`")
    parts.append(f"- Frequency CSV: `{paths.get('freq')}`\n")

    if stats_level is not None:
        parts.append("## Voltage summary")
        parts.append(_dict_to_markdown_table(stats_level))
        parts.append("")
        extras_v = extras.get("voltage", {})
        if extras_v:
            parts.append("### Voltage metrics")
            parts.append(_dict_to_markdown_table(extras_v))
            parts.append("")

    if stats_freq is not None:
        parts.append("## Frequency summary")
        parts.append(_dict_to_markdown_table(stats_freq))
        parts.append("")
        extras_f = extras.get("frequency", {})
        if extras_f:
            parts.append("### Frequency metrics")
            parts.append(_dict_to_markdown_table(extras_f))
            parts.append("")

    if filtered_extras is not None:
        parts.append("### Filtered frequency view (isolated outliers dropped)")
        parts.append(_dict_to_markdown_table(filtered_extras))
        parts.append("")

    if stats_level is not None and "mean" in stats_level:
        mean_v = stats_level.get("mean")
        rms_v = stats_level.get("rms")
        rng_v = (stats_level.get("min"), stats_level.get("max"))
        pct_ok = extras.get("voltage", {}).get("pct_within_ok_band")
        spikes = extras.get("voltage", {}).get("spike_count_over_135v")
        parts.append("## Voltage interpretation")
        parts.append(
            f"Average {mean_v:.2f} V with RMS {rms_v:.2f} V. "
            f"The range was {rng_v[0]:.2f} to {rng_v[1]:.2f} V. "
            f"About {pct_ok:.1f}% of samples were within {V_OK_MIN:.0f}–{V_OK_MAX:.0f} V. "
            f"Spike count (> {V_SPIKE:.0f} V): {int(spikes)}."
        )
        parts.append("")

    if stats_freq is not None and "median" in stats_freq:
        median_f = stats_freq.get("median")
        rng_f = (stats_freq.get("min"), stats_freq.get("max"))
        pct_t = extras.get("frequency", {}).get("pct_within_tight")
        pct_l = extras.get("frequency", {}).get("pct_within_loose")
        below = extras.get("frequency", {}).get("count_below_58hz")
        above = extras.get("frequency", {}).get("count_above_62hz")
        tail = ""
        if filtered_extras is not None:
            tail = " After dropping isolated outliers, frequency clusters remain close to 60 Hz."
        parts.append("## Frequency interpretation")
        parts.append(
            f"Median {median_f:.3f} Hz. Range {rng_f[0]:.3f} to {rng_f[1]:.3f} Hz. "
            f"{pct_t:.1f}% within ±{F_TIGHT:.2f} Hz and {pct_l:.1f}% within ±{F_LOOSE:.2f} Hz. "
            f"Counts outside [{F_HARD_MIN:.0f}, {F_HARD_MAX:.0f}] Hz: below={int(below)}, above={int(above)}.{tail}"
        )
        parts.append("")

    parts.append("## Plots")
    if "voltage_png" in figures:
        parts.append(f"![Voltage over time]({figures['voltage_png'].name})")
    if "frequency_png" in figures:
        parts.append(f"![Frequency over time]({figures['frequency_png'].name})")
    if "voltage_hist" in figures:
        parts.append(f"![Voltage distribution]({figures['voltage_hist'].name})")
    if "frequency_hist" in figures:
        parts.append(f"![Frequency distribution]({figures['frequency_hist'].name})")
    parts.append("")

    if footers:
        parts.append("## Device footers (as-recorded)")
        for kind, kv in footers.items():
            parts.append(f"### {kind.title()} footer")
            if kv:
                lines = [f"- `{k}` = `{v}`" for k, v in kv.items()]
                parts.extend(lines)
            else:
                parts.append("_(none)_")
            parts.append("")

    md = "\n".join(parts)
    if out_md_path:
        out_md_path.write_text(md, encoding="utf-8")
    return md


def write_summary_csv(d: Dict[str, object], path: Path) -> None:
    d2 = {}
    for k, v in d.items():
        if isinstance(v, datetime):
            d2[k] = v.strftime("%Y-%m-%d %H:%M:%S")
        else:
            d2[k] = v
    pd.DataFrame([d2]).to_csv(path, index=False)


def analyze_pair(
    base: str,
    level_path: Optional[Path],
    freq_path: Optional[Path],
    out_dir: Path,
    filter_freq: bool = False,
    make_hist: bool = False,
) -> Tuple[bool, Optional[Path]]:
    """
    Analyze a pair (or a single file if one is missing).
    Returns (success, report_path_or_None).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    have_any = False
    figures: Dict[str, Path] = {}
    extras: Dict[str, Dict[str, object]] = {}
    stats_level: Optional[Dict[str, object]] = None
    stats_freq: Optional[Dict[str, object]] = None
    filtered_extras: Optional[Dict[str, object]] = None
    footers: Dict[str, Dict[str, str]] = {}

    # Level (voltage)
    if level_path and level_path.exists():
        dfL = load_capture_csv(level_path, value_guess="level_v")
        stats_level = compute_basic_stats(dfL)
        extras["voltage"] = voltage_metrics(dfL, V_OK_MIN, V_OK_MAX, V_SPIKE)

        v_png = out_dir / f"{base}_voltage.png"
        plot_series(dfL, f"Voltage (V) over time — {base}", "Voltage (V)", v_png)
        figures["voltage_png"] = v_png

        if make_hist:
            v_hist = out_dir / f"{base}_voltage_hist.png"
            plot_hist(dfL, f"Voltage distribution — {base}", "Voltage (V)", v_hist)
            figures["voltage_hist"] = v_hist

        v_sum = out_dir / f"{base}_voltage_summary.csv"
        write_summary_csv({**stats_level, **extras["voltage"]}, v_sum)

        footers["voltage"] = parse_device_footers(level_path)
        have_any = True

    # Frequency
    if freq_path and freq_path.exists():
        dfF = load_capture_csv(freq_path, value_guess="freq_hz")
        dfF = dfF[dfF["value"] > 0].reset_index(drop=True)
        stats_freq = compute_basic_stats(dfF)
        extras["frequency"] = frequency_metrics(
            dfF, F_CENTER, F_TIGHT, F_LOOSE, F_HARD_MIN, F_HARD_MAX
        )

        if filter_freq and not dfF.empty:
            dfF_filt, filt_counts = apply_frequency_isolated_filter(
                dfF, F_HARD_MIN, F_HARD_MAX
            )
            stats_freq_filt = compute_basic_stats(dfF_filt)
            extras_filt = frequency_metrics(
                dfF_filt, F_CENTER, F_TIGHT, F_LOOSE, F_HARD_MIN, F_HARD_MAX
            )
            filtered_extras = {**stats_freq_filt, **extras_filt, **filt_counts}

        f_png = out_dir / f"{base}_frequency.png"
        plot_series(dfF, f"Frequency (Hz) over time — {base}", "Frequency (Hz)", f_png)
        figures["frequency_png"] = f_png

        if make_hist:
            f_hist = out_dir / f"{base}_frequency_hist.png"
            plot_hist(dfF, f"Frequency distribution — {base}", "Frequency (Hz)", f_hist)
            figures["frequency_hist"] = f_hist

        f_sum = out_dir / f"{base}_frequency_summary.csv"
        write_summary_csv({**stats_freq, **extras["frequency"]}, f_sum)

        footers["frequency"] = parse_device_footers(freq_path)
        have_any = True

    report_path = out_dir / f"{base}_report.md"
    render_report(
        base=base,
        paths={"level": level_path, "freq": freq_path},
        stats_level=stats_level,
        stats_freq=stats_freq,
        extras=extras,
        figures=figures,
        filtered_extras=filtered_extras,
        footers=footers,
        out_md_path=report_path,
    )

    return have_any, (report_path if have_any else None)


def find_pairs_in_dir(dir_path: Path) -> List[SeriesFiles]:
    """Discover files in a directory and return a list of SeriesFiles with possible pairs."""
    if not dir_path.exists() or not dir_path.is_dir():
        return []

    level_map: Dict[str, Path] = {}
    freq_map: Dict[str, Path] = {}

    for p in dir_path.iterdir():
        if p.suffix.lower() != ".csv":
            continue
        m = re.match(r"(\d{12})(F)?\.csv$", p.name, flags=re.IGNORECASE)
        if not m:
            continue
        base = m.group(1)
        is_freq = m.group(2)
        if is_freq:
            freq_map[base] = p
        else:
            level_map[base] = p

    bases = sorted(set(level_map.keys()) | set(freq_map.keys()))
    out: List[SeriesFiles] = []
    for b in bases:
        out.append(
            SeriesFiles(base=b, level_path=level_map.get(b), freq_path=freq_map.get(b))
        )
    return out


# ---------------------------
# CLI entry
# ---------------------------
def main_cli(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Analyze PowerSentry SD captures for voltage and frequency."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--dir",
        type=str,
        help="Directory containing capture CSVs (<BASE>.csv and <BASE>F.csv).",
    )
    g.add_argument("--level", type=str, help="Path to <BASE>.csv (voltage level).")
    ap.add_argument(
        "--freq",
        type=str,
        help="Path to <BASE>F.csv (frequency). Required if --level is used.",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for reports and figures.",
    )
    ap.add_argument(
        "--filter-freq",
        action="store_true",
        help="Enable filtered frequency view (drop isolated outliers).",
    )
    ap.add_argument("--hist", action="store_true", help="Also generate histograms.")
    args = ap.parse_args(argv)

    out_dir = Path(args.out).expanduser().resolve()
    any_success = False

    if args.dir:
        pairs = find_pairs_in_dir(Path(args.dir).expanduser().resolve())
        if not pairs:
            print("No capture files found in directory.")
        for sf in pairs:
            ok, _ = analyze_pair(
                sf.base,
                sf.level_path,
                sf.freq_path,
                out_dir,
                filter_freq=args.filter_freq,
                make_hist=args.hist,
            )
            if ok:
                any_success = True
    else:
        if not args.freq:
            ap.error("--freq is required when --level is used")
        level_path = Path(args.level).expanduser().resolve() if args.level else None
        freq_path = Path(args.freq).expanduser().resolve() if args.freq else None
        base = None
        if level_path:
            base = parse_base_from_filename(level_path)
        if not base and freq_path:
            base = parse_base_from_filename(freq_path)
        if not base:
            base = (level_path.stem if level_path else freq_path.stem).rstrip("F")
        ok, _ = analyze_pair(
            base,
            level_path,
            freq_path,
            out_dir,
            filter_freq=args.filter_freq,
            make_hist=args.hist,
        )
        if ok:
            any_success = True

    if not any_success:
        return 2
    return 0


# ---------------------------
# GUI
# ---------------------------
class AnalyzerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Power Spike Analyzer")
        self.geometry("920x600")
        self.minsize(820, 520)

        # State
        self.input_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.pairs: List[SeriesFiles] = []
        self.last_report: Optional[Path] = None

        # UI Vars
        self.var_input = tk.StringVar()
        self.var_output = tk.StringVar()
        self.var_filter = tk.BooleanVar(value=True)
        self.var_hist = tk.BooleanVar(value=False)

        self._build_widgets()

    # ---------- UI construction ----------
    def _build_widgets(self):
        pad = {"padx": 8, "pady": 6}

        top = ttk.Frame(self)
        top.pack(fill="x")

        # Input folder
        ttk.Label(top, text="SD folder:").grid(row=0, column=0, sticky="w", **pad)
        e_in = ttk.Entry(top, textvariable=self.var_input, width=70)
        e_in.grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(top, text="Browse…", command=self._choose_input).grid(
            row=0, column=2, **pad
        )

        # Output folder
        ttk.Label(top, text="Output folder:").grid(row=1, column=0, sticky="w", **pad)
        e_out = ttk.Entry(top, textvariable=self.var_output, width=70)
        e_out.grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(top, text="Browse…", command=self._choose_output).grid(
            row=1, column=2, **pad
        )

        # Options
        opts = ttk.Frame(self)
        opts.pack(fill="x")
        ttk.Checkbutton(
            opts, text="Filter freq (drop isolated outliers)", variable=self.var_filter
        ).pack(side="left", **pad)
        ttk.Checkbutton(opts, text="Generate histograms", variable=self.var_hist).pack(
            side="left", **pad
        )

        # Middle split: listbox + actions
        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True)

        # Listbox for pairs
        left = ttk.Frame(mid)
        left.pack(side="left", fill="both", expand=True, padx=8, pady=6)
        ttk.Label(left, text="Discovered captures:").pack(anchor="w")
        self.listbox = tk.Listbox(left, selectmode="browse")
        self.listbox.pack(fill="both", expand=True, pady=(0, 6))
        btns = ttk.Frame(left)
        btns.pack(fill="x")
        ttk.Button(btns, text="Scan Folder", command=self._scan).pack(
            side="left", padx=4
        )
        ttk.Button(btns, text="Analyze Selected", command=self._analyze_selected).pack(
            side="left", padx=4
        )
        ttk.Button(btns, text="Analyze All", command=self._analyze_all).pack(
            side="left", padx=4
        )

        # Right side controls
        right = ttk.Frame(mid)
        right.pack(side="left", fill="y", padx=8, pady=6)
        ttk.Button(right, text="Open Output Folder", command=self._open_output).pack(
            fill="x", pady=4
        )
        ttk.Button(right, text="View Last Report", command=self._open_last_report).pack(
            fill="x", pady=4
        )
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)
        ttk.Button(right, text="Quit", command=self.destroy).pack(fill="x", pady=4)

        # Log window
        logf = ttk.Frame(self)
        logf.pack(fill="both", expand=True)
        ttk.Label(logf, text="Log:").pack(anchor="w")
        self.log = ScrolledText(logf, height=10, wrap="word")
        self.log.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Make columns stretch
        top.grid_columnconfigure(1, weight=1)

    # ---------- helpers ----------
    def _log(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.update_idletasks()

    def _choose_input(self):
        path = filedialog.askdirectory(title="Select SD folder")
        if path:
            self.input_dir = Path(path)
            self.var_input.set(str(self.input_dir))
            # Default output to <input>/Reports if not set
            if not self.var_output.get():
                od = self.input_dir / "Reports"
                self.output_dir = od
                self.var_output.set(str(od))

    def _choose_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir = Path(path)
            self.var_output.set(str(self.output_dir))

    def _scan(self):
        self.listbox.delete(0, "end")
        self.pairs = []
        inp = self.var_input.get().strip()
        if not inp:
            messagebox.showwarning("Select folder", "Please choose an SD folder first.")
            return
        dirp = Path(inp).expanduser().resolve()
        if not dirp.exists():
            messagebox.showerror("Folder not found", f"Path does not exist:\n{dirp}")
            return
        self._log(f"Scanning {dirp}...")
        self.pairs = find_pairs_in_dir(dirp)
        if not self.pairs:
            self._log("No capture files found (*.csv / *F.csv).")
            return
        for sf in self.pairs:
            flags = []
            flags.append("L" if sf.level_path and sf.level_path.exists() else "-")
            flags.append("F" if sf.freq_path and sf.freq_path.exists() else "-")
            self.listbox.insert("end", f"{sf.base}   [{''.join(flags)}]")
        self._log(
            f"Found {len(self.pairs)} capture(s). Select one and click 'Analyze Selected' or use 'Analyze All'."
        )

    def _ensure_outdir(self) -> Optional[Path]:
        out = self.var_output.get().strip()
        if not out:
            messagebox.showwarning("Select output", "Please choose an output folder.")
            return None
        od = Path(out).expanduser().resolve()
        od.mkdir(parents=True, exist_ok=True)
        return od

    def _get_selected_pair(self) -> Optional[SeriesFiles]:
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo(
                "Select a capture", "Please select a capture in the list."
            )
            return None
        idx = sel[0]
        if idx < 0 or idx >= len(self.pairs):
            return None
        return self.pairs[idx]

    def _analyze_selected(self):
        sf = self._get_selected_pair()
        if not sf:
            return
        out_dir = self._ensure_outdir()
        if not out_dir:
            return
        self._log(f"Analyzing {sf.base}...")
        ok, rpt = analyze_pair(
            base=sf.base,
            level_path=sf.level_path,
            freq_path=sf.freq_path,
            out_dir=out_dir,
            filter_freq=self.var_filter.get(),
            make_hist=self.var_hist.get(),
        )
        if ok:
            self.last_report = rpt
            self._log(f"Done: {sf.base} → {rpt}")
        else:
            self._log(f"Skipped: {sf.base} (no analyzable files)")

    def _analyze_all(self):
        if not self.pairs:
            messagebox.showinfo("Nothing to analyze", "Scan a folder first.")
            return
        out_dir = self._ensure_outdir()
        if not out_dir:
            return
        processed: List[Tuple[str, Optional[Path]]] = []
        self._log(f"Analyzing all {len(self.pairs)} capture(s)...")
        cnt = 0
        for sf in self.pairs:
            ok, rpt = analyze_pair(
                base=sf.base,
                level_path=sf.level_path,
                freq_path=sf.freq_path,
                out_dir=out_dir,
                filter_freq=self.var_filter.get(),
                make_hist=self.var_hist.get(),
            )
            if ok:
                processed.append((sf.base, rpt))
                self.last_report = rpt
                cnt += 1
                self._log(f"  ✔ {sf.base}")
            else:
                self._log(f"  – {sf.base} (skipped)")
            self.update_idletasks()
        self._log(f"Processed {cnt} / {len(self.pairs)} capture(s).")
        if processed:
            self._write_index(processed, out_dir)

    def _write_index(self, processed: List[Tuple[str, Optional[Path]]], out_dir: Path):
        """Create a simple index.md linking to per-capture reports and plots."""
        lines = [
            "# PowerSentry Analysis Index",
            "",
            f"Total captures: {len(processed)}",
            "",
        ]
        for base, rpt in sorted(processed, key=lambda x: x[0]):
            vp = (out_dir / f"{base}_voltage.png").name
            fp = (out_dir / f"{base}_frequency.png").name
            lines.append(f"## {base}")
            if rpt:
                lines.append(f"- Report: [{rpt.name}]({rpt.name})")
            lines.append(f"- Plots: ![{vp}]({vp})  ![{fp}]({fp})")
            lines.append("")
        idx = out_dir / "index.md"
        idx.write_text("\n".join(lines), encoding="utf-8")
        self._log(f"Wrote index: {idx}")

    def _open_output(self):
        out = self.var_output.get().strip()
        if not out:
            messagebox.showinfo("No folder", "Output folder is not set.")
            return
        p = Path(out).expanduser().resolve()
        if sys.platform.startswith("win"):
            os.startfile(str(p))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])

    def _open_last_report(self):
        rpt = self.last_report
        if not rpt or not rpt.exists():
            messagebox.showinfo("No report", "No report has been generated yet.")
            return
        if sys.platform.startswith("win"):
            os.startfile(str(rpt))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(rpt)])
        else:
            subprocess.Popen(["xdg-open", str(rpt)])


def launch_gui():
    app = AnalyzerGUI()
    app.mainloop()


# ---------------------------
# Entrypoint selection
# ---------------------------
if __name__ == "__main__":
    # If any CLI flags are provided, run CLI. Else launch GUI.
    if len(sys.argv) > 1:
        sys.exit(main_cli())
    else:
        launch_gui()
