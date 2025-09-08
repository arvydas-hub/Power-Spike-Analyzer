#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Power Spike Analyzer — GUI + CLI (HTML reports, per-case folders, styled layout)

Fixes in this build:
  • Removed deprecated pandas `infer_datetime_format` usage and improved timestamp parsing.
  • Fixed TypeError in frequency_metrics (correctly compute 'above' count).
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
    """Parse time_local robustly without deprecated infer_datetime_format."""
    # Try explicit formats first; choose the one with the most non-nulls
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
    # Fallback to generic parser
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

    # Build timestamps
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
    above = int((v > f_hard_max).sum())  # <-- fixed
    return {
        "pct_within_tight": float(within_tight),
        "pct_within_loose": float(within_loose),
        "count_below_58hz": below,
        "count_above_62hz": above,
    }


def apply_frequency_isolated_filter(
    df: pd.DataFrame, f_hard_min: float, f_hard_max: float
) -> Tuple[pd.DataFrame, Dict[str, int]]:
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


# ---------- Scoring for traffic lights ----------
def _score_voltage(
    pct_ok: Optional[float], spikes: Optional[int]
) -> List[Tuple[str, str]]:
    tips = []
    if pct_ok is not None and not (isinstance(pct_ok, float) and math.isnan(pct_ok)):
        if pct_ok >= 95:
            color = "green"
            note = f"{_fmt_percent(pct_ok)} within 114–126 V"
        elif pct_ok >= 80:
            color = "yellow"
            note = f"{_fmt_percent(pct_ok)} within 114–126 V"
        else:
            color = "red"
            note = f"{_fmt_percent(pct_ok)} within 114–126 V"
        tips.append((color, note))
    if spikes is not None:
        if spikes == 0:
            tips.append(("green", "0 spikes over 135 V"))
        elif spikes <= 3:
            tips.append(("yellow", f"{spikes} spike(s) over 135 V"))
        else:
            tips.append(("red", f"{spikes} spikes over 135 V"))
    return tips


def _score_frequency(
    pct_tight: Optional[float], pct_loose: Optional[float], out_low: int, out_high: int
) -> List[Tuple[str, str]]:
    tips = []
    if pct_loose is not None and not (
        isinstance(pct_loose, float) and math.isnan(pct_loose)
    ):
        if pct_loose >= 99:
            tips.append(("green", f"{_fmt_percent(pct_loose)} within ±0.20 Hz"))
        elif pct_loose >= 95:
            tips.append(("yellow", f"{_fmt_percent(pct_loose)} within ±0.20 Hz"))
        else:
            tips.append(("red", f"{_fmt_percent(pct_loose)} within ±0.20 Hz"))
    if pct_tight is not None and not (
        isinstance(pct_tight, float) and math.isnan(pct_tight)
    ):
        if pct_tight >= 80:
            tips.append(("green", f"{_fmt_percent(pct_tight)} within ±0.05 Hz"))
        elif pct_tight >= 50:
            tips.append(("yellow", f"{_fmt_percent(pct_tight)} within ±0.05 Hz"))
        else:
            tips.append(("red", f"{_fmt_percent(pct_tight)} within ±0.05 Hz"))
    total_oob = (out_low or 0) + (out_high or 0)
    if total_oob == 0:
        tips.append(("green", "No samples outside 58–62 Hz"))
    elif total_oob <= 5:
        tips.append(("yellow", f"{total_oob} sample(s) outside 58–62 Hz"))
    else:
        tips.append(("red", f"{total_oob} samples outside 58–62 Hz"))
    return tips


def _badge(color: str, text: str) -> str:
    return f"<span class='dot {color}'></span>{text}"


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

    # Interpretations for hero
    v_interp = ""
    if stats_level is not None and "mean" in stats_level:
        mean_v = stats_level.get("mean")
        rms_v = stats_level.get("rms")
        rng_v = (stats_level.get("min"), stats_level.get("max"))
        pct_ok = extras.get("voltage", {}).get("pct_within_ok_band")
        spikes = extras.get("voltage", {}).get("spike_count_over_135v")
        badges = " &nbsp;•&nbsp; ".join(
            _badge(c, t) for c, t in _score_voltage(pct_ok, spikes)
        )
        v_interp = (
            f"<div class='line'><strong>Voltage:</strong> "
            f"Avg {_fmt_float(mean_v,'voltage')} V, RMS {_fmt_float(rms_v,'voltage')} V, "
            f"range {_fmt_float(rng_v[0],'voltage')}–{_fmt_float(rng_v[1],'voltage')} V.</div>"
            f"<div class='badges'>{badges}</div>"
        )

    f_interp = ""
    if stats_freq is not None and "median" in stats_freq:
        median_f = stats_freq.get("median")
        rng_f = (stats_freq.get("min"), stats_freq.get("max"))
        pct_t = extras.get("frequency", {}).get("pct_within_tight")
        pct_l = extras.get("frequency", {}).get("pct_within_loose")
        below = extras.get("frequency", {}).get("count_below_58hz")
        above = extras.get("frequency", {}).get("count_above_62hz")
        tail = (
            " After dropping isolated outliers, frequency clusters remain close to 60 Hz."
            if filtered_extras is not None
            else ""
        )
        badges = " &nbsp;•&nbsp; ".join(
            _badge(c, t)
            for c, t in _score_frequency(pct_t, pct_l, below or 0, above or 0)
        )
        f_interp = (
            f"<div class='line'><strong>Frequency:</strong> "
            f"Median {_fmt_float(median_f,'frequency')} Hz, "
            f"range {_fmt_float(rng_f[0],'frequency')}–{_fmt_float(rng_f[1],'frequency')} Hz.</div>"
            f"<div class='badges'>{badges}{h(tail)}</div>"
        )

    # --- HTML skeleton ---
    parts: List[str] = []
    parts.append(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>PowerSentry Capture Report — {h(base)}</title>"
        "<style>"
        ":root{--teal:#0f5f74;--teal-d:#0c4b5a;--yellow:#fff8cc;--yellow-b:#e0c300;"
        "--red:#ffe0e0;--red-b:#cc3333;--purple:#4c1d95;}"
        "body{font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif;line-height:1.45;"
        "max-width:1100px;margin:24px auto;padding:0 16px;}"
        "h1{margin:0 0 .2em 0;} h2{margin:1.0em 0 .4em;} h3{margin:.6em 0 .3em;}"
        ".hero{background:var(--teal);color:#fff;border-radius:14px;padding:16px 18px;box-shadow:0 2px 8px rgba(0,0,0,.08);}"
        ".hero .line{font-size:1.05rem;margin:.15em 0;}"
        ".hero .badges{opacity:.95;margin:.1em 0 .4em;}"
        ".dot{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle;margin:0 6px 2px 0;}"
        ".dot.green{background:#1e9e37;} .dot.yellow{background:#e2b100;} .dot.red{background:#d33;}"
        ".cards{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px;}"
        "@media(max-width: 900px){.cards{grid-template-columns:1fr;}}"
        ".card{border:1px solid #ccc;border-radius:14px;padding:12px 12px 10px;box-shadow:0 1px 6px rgba(0,0,0,.04);}"
        ".card.voltage{background:var(--yellow);border-color:var(--yellow-b);}"
        ".card.frequency{background:var(--red);border-color:var(--red-b);}"
        ".sgrid{display:grid;grid-template-columns: 1fr 1fr;grid-auto-rows:auto;gap:10px;align-items:start;}"
        ".sgrid .plot{grid-column:1 / span 2;}"
        ".kv{border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;}"
        ".kv td,.kv th{border:1px solid #ddd;padding:4px 8px;} .kv th{background:#f2f2f2;font-weight:600;}"
        "img{max-width:100%;height:auto;border:1px solid #eee;padding:2px;background:#fff;border-radius:8px;}"
        ".band-bottom{background:var(--purple);color:#fff;border-radius:14px;padding:14px;margin-top:14px;}"
        ".band-bottom h2{color:#fff;margin-top:0;}"
        ".band-bottom .inner{background:rgba(255,255,255,.08);padding:8px;border-radius:10px;}"
        ".files code{background:#f5f5f5;padding:1px 4px;border-radius:3px;}"
        "</style></head><body>"
    )

    parts.append("<div class='hero'>")
    parts.append(f"<h1>PowerSentry Capture Report — {h(base)}</h1>")
    if v_interp:
        parts.append(v_interp)
    if f_interp:
        parts.append(f_interp)
    parts.append("</div>")

    # Cards: voltage + frequency
    parts.append("<div class='cards'>")

    if stats_level is not None:
        parts.append("<div class='card voltage'>")
        parts.append("<h2>Voltage</h2>")
        parts.append("<div class='sgrid'>")
        parts.append(_dict_to_html_table(stats_level, "Summary"))
        extras_v = extras.get("voltage", {})
        if extras_v:
            parts.append(_dict_to_html_table(extras_v, "Metrics"))
        else:
            parts.append("<div></div>")
        if "voltage_png" in figures:
            parts.append(
                f"<div class='plot'><img alt='Voltage over time' src='{figures['voltage_png'].name}'></div>"
            )
        parts.append("</div></div>")

    if stats_freq is not None:
        parts.append("<div class='card frequency'>")
        parts.append("<h2>Frequency</h2>")
        parts.append("<div class='sgrid'>")
        parts.append(_dict_to_html_table(stats_freq, "Summary"))
        if filtered_extras is not None:
            parts.append(_dict_to_html_table(filtered_extras, "Filtered metrics"))
        else:
            extras_f = extras.get("frequency", {})
            if extras_f:
                parts.append(_dict_to_html_table(extras_f, "Metrics"))
            else:
                parts.append("<div></div>")
        if "frequency_png" in figures:
            parts.append(
                f"<div class='plot'><img alt='Frequency over time' src='{figures['frequency_png'].name}'></div>"
            )
        parts.append("</div></div>")
    parts.append("</div>")  # .cards

    # Bottom band (purple): distributions + files + footers
    show_hist = ("voltage_hist" in figures) or ("frequency_hist" in figures)
    show_footers = footers and any(bool(kv) for kv in footers.values())
    if show_hist or paths or show_footers:
        parts.append("<div class='band-bottom'>")
        if show_hist:
            parts.append("<h2>Distributions</h2><div class='inner'>")
            if "voltage_hist" in figures:
                parts.append(
                    f"<div><img alt='Voltage distribution' src='{figures['voltage_hist'].name}'></div>"
                )
            if "frequency_hist" in figures:
                parts.append(
                    f"<div><img alt='Frequency distribution' src='{figures['frequency_hist'].name}'></div>"
                )
            parts.append("</div>")
        if paths:
            parts.append("<h2>Files</h2><div class='inner'><ul class='files'>")
            parts.append(
                f"<li>Level CSV: <code>{h(str(paths.get('level') or ''))}</code></li>"
            )
            parts.append(
                f"<li>Frequency CSV: <code>{h(str(paths.get('freq') or ''))}</code></li>"
            )
            parts.append("</ul></div>")
        if show_footers:
            parts.append("<h2>Device footers (as-recorded)</h2><div class='inner'>")
            for kind, kv in footers.items():
                if not kv:
                    continue
                parts.append(f"<h3>{kind.title()} footer</h3><ul>")
                for k, v in kv.items():
                    parts.append(f"<li><code>{h(k)}</code> = <code>{h(v)}</code></li>")
                parts.append("</ul>")
            parts.append("</div>")
        parts.append("</div>")  # .band-bottom

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
    # Per-case subfolder
    case_dir = out_dir / base
    case_dir.mkdir(parents=True, exist_ok=True)

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
        figures={
            k: Path(v.name) for k, v in figures.items()
        },  # images referenced by filename only
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
        ttk.Button(right, text="View Last Report", command=self._open_last_report).pack(
            fill="x", pady=4
        )
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

    def _open_last_report(self):
        rpt = self.last_report
        if rpt and rpt.exists():
            if sys.platform.startswith("win"):
                os.startfile(str(rpt))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(rpt)])
            else:
                subprocess.Popen(["xdg-open", str(rpt)])
            return

        res = messagebox.askyesno(
            "No report found", "No report found. Generate one for the selected capture?"
        )
        if not res:
            return
        sf = self._get_selected_pair()
        if not sf:
            messagebox.showinfo(
                "Select a capture",
                "Please select a capture in the list, then try again.",
            )
            return
        out_dir = self._ensure_outdir()
        if not out_dir:
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
        if ok:
            self.last_report = rpt2
            self._log(f"Done: {sf.base} → {rpt2}")
            if sys.platform.startswith("win"):
                os.startfile(str(rpt2))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(rpt2)])
            else:
                subprocess.Popen(["xdg-open", str(rpt2)])
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
