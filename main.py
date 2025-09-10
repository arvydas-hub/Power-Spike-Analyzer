#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Power Spike Analyzer — GUI + CLI (HTML reports, per-case folders)
# Design: Gemini-style report (hero + aligned cards), colored status dots, per-case subfolders.
# Fixes: robust datetime parsing; correct frequency counts; "Open Selected Report" button.

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
import pandas as pd

matplotlib.use("Agg")
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
    m = re.search(r"(\d{12})(?:F)?\.csv$", p.name, flags=re.IGNORECASE)
    return m.group(1) if m else None


def parse_base_ts(base: str) -> Optional[datetime]:
    try:
        return datetime.strptime(base, "%m%d%y%H%M%S")
    except Exception:
        return None


def _parse_time_local(col: pd.Series) -> pd.Series:
    """Parse time_local robustly (no deprecated infer_datetime_format)."""
    fmts = ["%m/%d/%y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%m-%d-%y %H:%M:%S"]
    best = None
    best_valid = -1
    for fmt in fmts:
        try:
            parsed = pd.to_datetime(col, format=fmt, errors="coerce")
        except Exception:
            parsed = None
        if parsed is not None:
            valid = parsed.notna().sum()
            if valid > best_valid:
                best = parsed
                best_valid = valid
    if best is not None and best_valid > 0:
        return best
    return pd.to_datetime(col, errors="coerce")


def load_capture_csv(path: Path, value_guess: Optional[str] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, comment="#", dtype=str).copy()
    if df.shape[0] == 0 and df.shape[1] == 0:
        return pd.DataFrame(columns=["ts", "value"])

    df.columns = _norm_cols(list(df.columns))

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

    # timestamps
    if "time_local" in df.columns:
        ts = _parse_time_local(df["time_local"])
    else:
        ts = None

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


# ---------- Formatting helpers ----------
def _fmt_percent(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{x:.1f}%"


def _fmt_float(x: float, kind: str = "generic") -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    if kind == "frequency":
        return f"{x:.2f}"
    if kind == "voltage":
        return f"{x:.1f}"
    if kind == "duration":
        return f"{x:.0f}"
    return f"{x:.1f}"


def _fmt_value_for_key(val, key: str) -> str:
    if key.startswith("pct_"):
        return _fmt_percent(float(val))
    if key in (
        "count",
        "spike_count_over_135v",
        "count_below_58hz",
        "count_above_62hz",
        "total_out_of_band",
        "dropped_isolated_singletons",
    ):
        try:
            return str(int(val))
        except Exception:
            return ""
    if key in ("duration_s",):
        return _fmt_float(float(val), "duration")
    if key in ("min", "max", "mean", "median", "rms", "std_pop"):
        try:
            fv = float(val)
        except Exception:
            return ""
        if 40.0 < fv < 80.0:
            return _fmt_float(fv, "frequency")
        else:
            return _fmt_float(fv, "voltage")
    try:
        return _fmt_float(float(val))
    except Exception:
        return str(val)


def _dict_to_html_table(d: Dict[str, object], title: Optional[str] = None) -> str:
    keys_in = list(d.keys())
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
        "pct_within_ok_band",
        "spike_count_over_135v",
        "pct_within_tight",
        "pct_within_loose",
        "count_below_58hz",
        "count_above_62hz",
        "total_out_of_band",
        "dropped_isolated_singletons",
    ]
    ordered = [k for k in preferred if k in d] + [
        k for k in keys_in if k not in preferred
    ]

    header = f"<h3>{title}</h3>" if title else ""
    rows = []
    for k in ordered:
        v = d.get(k, "")
        if isinstance(v, datetime):
            s = v.strftime("%Y-%m-%d %H:%M:%S")
        else:
            s = _fmt_value_for_key(v, k)
        rows.append(f"<tr><td>{k}</td><td>{s}</td></tr>")
    table = (
        "<table class='kv'><thead><tr><th>metric</th><th>value</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )
    return header + table


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
        return {"pct_within_ok_band": math.nan, "spike_count_over_135v": 0}
    v = df["value"].astype(float)
    within = ((v >= v_ok_min) & (v <= v_ok_max)).mean() * 100.0
    spike_count = int((v > v_spike).sum())
    return {"pct_within_ok_band": float(within), "spike_count_over_135v": spike_count}


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
):
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
    else:
        plt.figure()
        plt.hist(df["value"].astype(float).dropna(), bins=bins)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png_path)
    plt.close()


# ---------- Status / badges ----------
def _status_voltage(pct_ok: Optional[float], spikes: Optional[int]) -> str:
    if pct_ok is None or (isinstance(pct_ok, float) and math.isnan(pct_ok)):
        return "warning"
    if pct_ok >= 95 and (spikes or 0) == 0:
        return "ok"
    if pct_ok >= 80 and (spikes or 0) <= 3:
        return "warning"
    return "issue"


def _status_frequency(pct_loose: Optional[float], out_low: int, out_high: int) -> str:
    total = (out_low or 0) + (out_high or 0)
    if pct_loose is None or (isinstance(pct_loose, float) and math.isnan(pct_loose)):
        return "warning"
    if pct_loose >= 99 and total == 0:
        return "ok"
    if pct_loose >= 95 and total <= 5:
        return "warning"
    return "issue"


def render_report_html(
    base: str,
    paths: Dict[str, Optional[Path]],
    stats_level: Optional[Dict[str, object]],
    stats_freq: Optional[Dict[str, object]],
    extras: Dict[str, Dict[str, object]],
    figures: Dict[str, Path],
    filtered_extras: Optional[Dict[str, object]] = None,
    footers: Optional[Dict[str, Dict[str, str]]] = None,
    out_html_path: Optional[Path] = None,
) -> str:
    def h(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Build hero summary lines
    v_line = ""
    f_line = ""
    v_sub = ""
    f_sub = ""
    v_dot = "yellow"
    f_dot = "yellow"

    if stats_level:
        mean_v = stats_level.get("mean")
        rms_v = stats_level.get("rms")
        rng_v = (stats_level.get("min"), stats_level.get("max"))
        pct_ok = extras.get("voltage", {}).get("pct_within_ok_band")
        spikes = extras.get("voltage", {}).get("spike_count_over_135v")
        v_line = f"<strong>Voltage:</strong> Avg {_fmt_float(mean_v,'voltage')} V, RMS {_fmt_float(rms_v,'voltage')} V, range {_fmt_float(rng_v[0],'voltage')}–{_fmt_float(rng_v[1],'voltage')} V."
        v_sub = f"{_fmt_percent(pct_ok)} within {V_OK_MIN:.0f}–{V_OK_MAX:.0f} V &bull; {int(spikes or 0)} spikes over {V_SPIKE:.0f} V"
        v_dot = {"ok": "green", "warning": "yellow", "issue": "red"}[
            _status_voltage(pct_ok, spikes)
        ]

    if stats_freq:
        median_f = stats_freq.get("median")
        rng_f = (stats_freq.get("min"), stats_freq.get("max"))
        pct_t = extras.get("frequency", {}).get("pct_within_tight")
        pct_l = extras.get("frequency", {}).get("pct_within_loose")
        below = int(extras.get("frequency", {}).get("count_below_58hz") or 0)
        above = int(extras.get("frequency", {}).get("count_above_62hz") or 0)
        f_line = f"<strong>Frequency:</strong> Median {_fmt_float(median_f,'frequency')} Hz, range {_fmt_float(rng_f[0],'frequency')}–{_fmt_float(rng_f[1],'frequency')} Hz."
        f_sub = f"{_fmt_percent(pct_l)} within ±{F_LOOSE:.2f} Hz &bull; {_fmt_percent(pct_t)} within ±{F_TIGHT:.2f} Hz &bull; {(below+above)} sample(s) outside {F_HARD_MIN:.0f}–{F_HARD_MAX:.0f} Hz"
        if filtered_extras is not None:
            f_sub += " <br>After dropping isolated outliers, frequency clusters remain close to 60 Hz."
        f_dot = {"ok": "green", "warning": "yellow", "issue": "red"}[
            _status_frequency(pct_l, below, above)
        ]

    # --- HTML skeleton ---
    parts: List[str] = []
    parts.append(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>PowerSentry Capture Report — {h(base)}</title>"
        "<style>"
        ":root{--bg-color:#ffffff;--section-bg:#f8fafc;--header-bg:#1e293b;--text-color:#1e293b;"
        "--text-light:#f8fafc;--text-muted:#64748b;--border-color:#e2e8f0;"
        "--shadow:0 4px 6px -1px rgba(0,0,0,.07),0 2px 4px -2px rgba(0,0,0,.07);"
        "--ok-bg:#f0fdf4;--ok-border:#bbf7d0;--ok-text:#166534;--ok-dot:#22c55e;"
        "--warning-bg:#fffbeb;--warning-border:#fde68a;--warning-text:#854d0e;--warning-dot:#f59e0b;"
        "--issue-bg:#fef2f2;--issue-border:#fecaca;--issue-text:#991b1b;--issue-dot:#ef4444;}"
        "body{font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif;line-height:1.6;"
        "background-color:var(--bg-color);color:var(--text-color);max-width:1200px;margin:2rem auto;padding:0 1.5rem;}"
        "h1,h2,h3{line-height:1.2;letter-spacing:-.02em;} h1{font-size:1.8rem;margin:0;}"
        "h2{font-size:1.5rem;margin:2rem 0 1rem;border-bottom:1px solid var(--border-color);padding-bottom:.5rem;}"
        "h3{font-size:1.1rem;margin:1.2rem 0 .5rem;}"
        ".hero{background:var(--header-bg);color:var(--text-light);border-radius:12px;padding:1.5rem 2rem;margin-bottom:2rem;box-shadow:var(--shadow);}"
        ".hero .line{font-size:1.05rem;margin:.25em 0;opacity:.9;}"
        ".hero-summary{margin-top:1.1rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,.2);display:grid;gap:1rem;}"
        ".summary-item{display:flex;align-items:flex-start;gap:.8rem;} .summary-item strong{font-weight:600;}"
        ".sub-line{font-size:.92rem;color:rgba(255,255,255,.9);margin-top:.25rem;}"
        ".dot{display:inline-block;width:11px;height:11px;border-radius:50%;flex-shrink:0;margin-top:6px;}"
        ".dot.green{background:var(--ok-dot);} .dot.yellow{background:var(--warning-dot);} .dot.red{background:var(--issue-dot);}"
        ".band{background:var(--section-bg);border:1px solid var(--border-color);border-radius:12px;padding:1.2rem;"
        "margin-bottom:1.2rem;overflow:hidden;}"
        ".band.ok{background:var(--ok-bg);border-color:var(--ok-border);} .band.ok h3{color:var(--ok-text);}"
        ".band.warning{background:var(--warning-bg);border-color:var(--warning-border);} .band.warning h3{color:var(--warning-text);}"
        ".band.issue{background:var(--issue-bg);border-color:var(--issue-border);} .band.issue h3{color:var(--issue-text);}"
        ".content-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1.2rem;align-items:flex-start;}"
        "table.kv{width:100%;border-collapse:collapse;font-size:.95rem;background-color:var(--bg-color);border-radius:8px;"
        "box-shadow:var(--shadow);border:1px solid var(--border-color);}"
        "table.kv th,table.kv td{border-bottom:1px solid var(--border-color);padding:.6rem .8rem;text-align:left;}"
        "table.kv thead{display:none;} table.kv tr:last-child td{border-bottom:none;} table.kv td:first-child{font-weight:500;}"
        ".plot{margin-top:.5rem;} .plot img{display:block;max-width:100%;height:auto;border-radius:8px;border:1px solid var(--border-color);background:#fff;}"
        ".files code{background:#f1f5f9;padding:2px 6px;border-radius:6px;}"
        "</style></head><body>"
    )

    parts.append("<div class='hero'>")
    parts.append("<h1>PowerSentry Capture Report</h1>")
    parts.append(f"<div class='line'>Capture ID: {h(base)}</div>")
    parts.append("<div class='hero-summary'>")
    if v_line:
        parts.append(
            f"<div class='summary-item'><span class='dot {v_dot}'></span><div>{v_line}<div class='sub-line'>{v_sub}</div></div></div>"
        )
    if f_line:
        parts.append(
            f"<div class='summary-item'><span class='dot {f_dot}'></span><div>{f_line}<div class='sub-line'>{f_sub}</div></div></div>"
        )
    parts.append("</div></div>")  # hero

    # Voltage section
    if stats_level is not None:
        pct_ok = extras.get("voltage", {}).get("pct_within_ok_band")
        spikes = extras.get("voltage", {}).get("spike_count_over_135v")
        v_status = {"ok": "ok", "warning": "warning", "issue": "issue"}[
            _status_voltage(pct_ok, spikes)
        ]
        parts.append("<h2>Voltage Analysis</h2>")
        parts.append(f"<div class='band {v_status}'>")
        parts.append("<div class='content-grid'>")
        merged = {**stats_level, **(extras.get("voltage", {}) or {})}
        parts.append("<div>")
        parts.append("<h3>Metrics</h3>")
        parts.append(_dict_to_html_table(merged))
        parts.append("</div>")
        if "voltage_png" in figures:
            parts.append("<div class='plot'>")
            parts.append(
                f"<img alt='Voltage over time' src='{figures['voltage_png'].name}'>"
            )
            parts.append("</div>")
        parts.append("</div></div>")

    # Frequency section
    if stats_freq is not None:
        extras_f = extras.get("frequency", {}) or {}
        below = int(extras_f.get("count_below_58hz") or 0)
        above = int(extras_f.get("count_above_62hz") or 0)
        f_status = {"ok": "ok", "warning": "warning", "issue": "issue"}[
            _status_frequency(extras_f.get("pct_within_loose"), below, above)
        ]
        parts.append("<h2>Frequency Analysis</h2>")
        parts.append(f"<div class='band {f_status}'>")
        parts.append("<div class='content-grid'>")
        parts.append("<div>")
        parts.append("<h3>Raw Metrics</h3>")
        parts.append(_dict_to_html_table(stats_freq))
        parts.append("</div>")
        parts.append("<div>")
        if filtered_extras is not None:
            parts.append("<h3>Filtered Metrics</h3>")
            parts.append(_dict_to_html_table(filtered_extras))
        elif extras_f:
            parts.append("<h3>Metrics</h3>")
            parts.append(_dict_to_html_table(extras_f))
        parts.append("</div>")
        parts.append("</div>")
        if "frequency_png" in figures:
            parts.append("<div class='plot'>")
            parts.append(
                f"<img alt='Frequency over time' src='{figures['frequency_png'].name}'>"
            )
            parts.append("</div>")
        parts.append("</div>")

    # Distributions (if any)
    if "voltage_hist" in figures or "frequency_hist" in figures:
        parts.append("<h2>Distributions</h2>")
        parts.append("<div class='band'>")
        parts.append("<div class='content-grid'>")
        if "voltage_hist" in figures:
            parts.append("<div class='plot'>")
            parts.append(
                f"<img alt='Voltage distribution' src='{figures['voltage_hist'].name}'>"
            )
            parts.append("</div>")
        if "frequency_hist" in figures:
            parts.append("<div class='plot'>")
            parts.append(
                f"<img alt='Frequency distribution' src='{figures['frequency_hist'].name}'>"
            )
            parts.append("</div>")
        parts.append("</div></div>")

    # Files band
    parts.append("<h2>Files</h2>")
    parts.append("<div class='band'><ul>")
    parts.append(f"<li>Level CSV: <code>{h(str(paths.get('level') or ''))}</code></li>")
    parts.append(
        f"<li>Frequency CSV: <code>{h(str(paths.get('freq') or ''))}</code></li>"
    )
    parts.append("</ul></div>")

    parts.append("</body></html>")
    html = "\n".join(parts)
    if out_html_path:
        out_html_path.write_text(html, encoding="utf-8")
    return html


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
    case_dir = out_dir / base
    case_dir.mkdir(parents=True, exist_ok=True)

    have_any = False
    figures: Dict[str, Path] = {}
    extras: Dict[str, Dict[str, object]] = {}
    stats_level: Optional[Dict[str, object]] = None
    stats_freq: Optional[Dict[str, object]] = None
    filtered_extras: Optional[Dict[str, object]] = None
    footers: Dict[str, Dict[str, str]] = {}

    if level_path and level_path.exists():
        dfL = load_capture_csv(level_path, value_guess="level_v")
        stats_level = compute_basic_stats(dfL)
        extras["voltage"] = voltage_metrics(dfL, V_OK_MIN, V_OK_MAX, V_SPIKE)

        v_png = case_dir / f"{base}_voltage.png"
        plot_series(dfL, f"Voltage (V) over time — {base}", "Voltage (V)", v_png)
        figures["voltage_png"] = v_png

        if make_hist:
            v_hist = case_dir / f"{base}_voltage_hist.png"
            plot_hist(dfL, f"Voltage distribution — {base}", "Voltage (V)", v_hist)
            figures["voltage_hist"] = v_hist

        v_sum = case_dir / f"{base}_voltage_summary.csv"
        write_summary_csv({**stats_level, **extras["voltage"]}, v_sum)

        footers["voltage"] = parse_device_footers(level_path)
        have_any = True

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

        f_png = case_dir / f"{base}_frequency.png"
        plot_series(dfF, f"Frequency (Hz) over time — {base}", "Frequency (Hz)", f_png)
        figures["frequency_png"] = f_png

        if make_hist:
            f_hist = case_dir / f"{base}_frequency_hist.png"
            plot_hist(dfF, f"Frequency distribution — {base}", "Frequency (Hz)", f_hist)
            figures["frequency_hist"] = f_hist

        f_sum = case_dir / f"{base}_frequency_summary.csv"
        write_summary_csv({**stats_freq, **extras["frequency"]}, f_sum)

        footers["frequency"] = parse_device_footers(freq_path)
        have_any = True

    report_path = case_dir / f"{base}_report.html"
    render_report_html(
        base=base,
        paths={"level": level_path, "freq": freq_path},
        stats_level=stats_level,
        stats_freq=stats_freq,
        extras=extras,
        figures={k: Path(v.name) for k, v in figures.items()},
        filtered_extras=filtered_extras,
        footers=footers,
        out_html_path=report_path,
    )

    return have_any, (report_path if have_any else None)


def find_pairs_in_dir(dir_path: Path) -> List[SeriesFiles]:
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
# CLI
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

        self.input_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.pairs: List[SeriesFiles] = []
        self.last_report: Optional[Path] = None

        self.var_input = tk.StringVar()
        self.var_output = tk.StringVar()
        self.var_filter = tk.BooleanVar(value=True)
        self.var_hist = tk.BooleanVar(value=False)

        self._build_widgets()

    def _build_widgets(self):
        pad = {"padx": 8, "pady": 6}

        top = ttk.Frame(self)
        top.pack(fill="x")

        ttk.Label(top, text="SD folder:").grid(row=0, column=0, sticky="w", **pad)
        e_in = ttk.Entry(top, textvariable=self.var_input, width=70)
        e_in.grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(top, text="Browse…", command=self._choose_input).grid(
            row=0, column=2, **pad
        )

        ttk.Label(top, text="Output folder:").grid(row=1, column=0, sticky="w", **pad)
        e_out = ttk.Entry(top, textvariable=self.var_output, width=70)
        e_out.grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(top, text="Browse…", command=self._choose_output).grid(
            row=1, column=2, **pad
        )

        opts = ttk.Frame(self)
        opts.pack(fill="x")
        ttk.Checkbutton(
            opts, text="Filter freq (drop isolated outliers)", variable=self.var_filter
        ).pack(side="left", **pad)
        ttk.Checkbutton(opts, text="Generate histograms", variable=self.var_hist).pack(
            side="left", **pad
        )

        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True)

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

        right = ttk.Frame(mid)
        right.pack(side="left", fill="y", padx=8, pady=6)
        ttk.Button(right, text="Open Output Folder", command=self._open_output).pack(
            fill="x", pady=4
        )
        ttk.Button(
            right, text="Open Selected Report", command=self._open_selected_report
        ).pack(fill="x", pady=4)
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)
        ttk.Button(right, text="Quit", command=self.destroy).pack(fill="x", pady=4)

        logf = ttk.Frame(self)
        logf.pack(fill="both", expand=True)
        ttk.Label(logf, text="Log:").pack(anchor="w")
        self.log = ScrolledText(logf, height=10, wrap="word")
        self.log.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        top.grid_columnconfigure(1, weight=1)

    def _log(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.update_idletasks()

    def _choose_input(self):
        path = filedialog.askdirectory(title="Select SD folder")
        if path:
            self.input_dir = Path(path)
            self.var_input.set(str(self.input_dir))
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
            return None
        idx = sel[0]
        if idx < 0 or idx >= len(self.pairs):
            return None
        return self.pairs[idx]

    def _analyze_selected(self):
        sf = self._get_selected_pair()
        if not sf:
            messagebox.showinfo(
                "Select a capture", "Please select a capture in the list."
            )
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
        html = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>PowerSentry Analysis Index</title>"
            "<style>body{font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif;"
            "line-height:1.45;max-width:980px;margin:24px auto;padding:0 16px;}"
            "a{color:#0645ad;text-decoration:none;}a:hover{text-decoration:underline;}"
            "img{max-width:100%;height:auto;border:1px solid #eee;padding:2px;background:#fff;}"
            "h1,h2{margin-top:1.2em;}</style></head><body>"
        ]
        html.append("<h1>PowerSentry Analysis Index</h1>")
        html.append(f"<p>Total captures: {len(processed)}</p>")
        for base, rpt in sorted(processed, key=lambda x: x[0]):
            sub = f"{base}/"
            rpt_href = f"{sub}{base}_report.html" if rpt else ""
            vp = f"{sub}{base}_voltage.png"
            fp = f"{sub}{base}_frequency.png"
            html.append(f"<h2>{base}</h2>")
            if rpt:
                html.append(f"<p>Report: <a href='{rpt_href}'>{rpt_href}</a></p>")
            html.append(
                f"<p>Plots:<br><img alt='Voltage {base}' src='{vp}'> "
                f"<img alt='Frequency {base}' src='{fp}'></p>"
            )
        html.append("</body></html>")
        idx = out_dir / "index.html"
        idx.write_text("\n".join(html), encoding="utf-8")
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

    def _open_selected_report(self):
        # Open report for the currently selected capture; prompt to generate if missing.
        sf = self._get_selected_pair()
        if not sf:
            messagebox.showinfo(
                "Select a capture", "Please select a capture in the list."
            )
            return
        out_dir = self._ensure_outdir()
        if not out_dir:
            return
        rpt_path = out_dir / sf.base / f"{sf.base}_report.html"
        if rpt_path.exists():
            try:
                if sys.platform.startswith("win"):
                    os.startfile(str(rpt_path))
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(rpt_path)])
                else:
                    subprocess.Popen(["xdg-open", str(rpt_path)])
            except Exception as e:
                messagebox.showerror(
                    "Open failed", f"Could not open report:\n{rpt_path}\n\n{e}"
                )
            return

        # Not found — ask to generate via the same routine as "Analyze Selected"
        res = messagebox.askyesno(
            "Report not found",
            "No report found for the selected capture. Generate one now?",
        )
        if not res:
            return

        self._log(f"Analyzing {sf.base}...")
        ok, rpt2 = analyze_pair(
            base=sf.base,
            level_path=sf.level_path,
            freq_path=sf.freq_path,
            out_dir=out_dir,
            filter_freq=self.var_filter.get(),
            make_hist=self.var_hist.get(),
        )
        if ok and rpt2 and rpt2.exists():
            self.last_report = rpt2
            self._log(f"Done: {sf.base} → {rpt2}")
            try:
                if sys.platform.startswith("win"):
                    os.startfile(str(rpt2))
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(rpt2)])
                else:
                    subprocess.Popen(["xdg-open", str(rpt2)])
            except Exception as e:
                messagebox.showerror(
                    "Open failed", f"Could not open report:\n{rpt2}\n\n{e}"
                )
        else:
            self._log("No analyzable files in the selected capture.")


def launch_gui():
    app = AnalyzerGUI()
    app.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main_cli())
    else:
        launch_gui()
