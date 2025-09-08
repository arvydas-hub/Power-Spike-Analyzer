#!/usr/bin/env python3
"""
PowerSentry SD Analyzer — ps_analyze.py

Analyzes M5Stack Core2 PowerSentry SD-card captures of AC voltage "level" and frequency.
Implements the spec agreed with Arvy on 2025-09-08.

Inputs and file conventions
---------------------------
Accept either:
  • Two explicit file paths: <BASE>.csv (voltage level) and <BASE>F.csv (frequency), or
  • A directory path to scan for pairs. The base name is MMDDYYhhmmss.
    The level file is <BASE>.csv. The frequency file is <BASE>F.csv.

CSV details:
  • Comment lines start with '#' and should be ignored.
  • Expected columns per row (case-insensitive, whitespace-tolerant):
      - time_local (string, local timestamp)
      - ms (integer, optional)
      - level_v (float)  for voltage files
      - freq_hz (float)  for frequency files
  • Backward compatibility:
      If time_local is missing, derive a base timestamp from the filename <BASE> interpreted
      as MMDDYYhhmmss. Use ms to offset inside the event. If ms is missing, treat as zero.

Computations
------------
For each series (level and frequency):
  - samples count
  - min, max, mean, median, stddev (population)
  - rms
  - start, end, and duration_s

Additional metrics for voltage:
  - Percent of samples within [114.0, 126.0] V (ANSI C84.1 Range A surrogate)
  - Count of samples > 135.0 V ("spike count")

Additional metrics for frequency:
  - Percent within 60.00 ± 0.05 Hz
  - Percent within 60.00 ± 0.20 Hz
  - Count of samples < 58.0 Hz and > 62.0 Hz
  - Optional filtered view to mimic device v3.0.6 logic:
      Mark a sample as out-of-band if value < 58.0 or value > 62.0
      Drop only isolated out-of-band singletons (keep clusters)
      Provide filtered-series metrics and counts:
        • dropped_out_of_band_total
        • dropped_isolated_singletons

Visuals
-------
  - Time series line chart for Voltage over time
  - Time series line chart for Frequency over time
  - Optional histograms (via --hist)

CLI
---
  Analyze a pair:
    python ps_analyze.py --level <path/to/BASE.csv> --freq <path/to/BASEF.csv> --out <report_dir> [--filter-freq] [--hist]

  Analyze a directory (find pairs):
    python ps_analyze.py --dir <sd_dump_dir> --out <report_dir> [--filter-freq] [--hist]

Outputs
-------
  <BASE>_voltage_summary.csv
  <BASE>_frequency_summary.csv
  <BASE>_voltage.png
  <BASE>_frequency.png
  <BASE>_report.md

Requirements
------------
  Python 3.9+ recommended
  pip install pandas matplotlib
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

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
    """
    Extract 12-digit base from filenames like 090725051139.csv or 090725051139F.csv.
    Returns None if not matched.
    """
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

    # Read CSV with comment lines skipped
    df = pd.read_csv(path, comment="#", dtype=str).copy()
    if df.shape[0] == 0 and df.shape[1] == 0:
        # Empty after skipping comments
        return pd.DataFrame(columns=["ts", "value"])

    df.columns = _norm_cols(list(df.columns))

    # Identify value column
    cols = set(df.columns)
    candidates_ordered = []
    if value_guess:
        candidates_ordered.append(value_guess.strip().lower())

    # Add common synonyms
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
        # Try to heuristically guess: pick first numeric-looking column
        for c in df.columns:
            try:
                pd.to_numeric(df[c])
                value_col = c
                break
            except Exception:
                continue

    if value_col is None:
        # No usable value column
        return pd.DataFrame(columns=["ts", "value"])

    # Parse value as float
    df["value"] = pd.to_numeric(df[value_col], errors="coerce")

    # Build timestamps
    base = parse_base_from_filename(path) or ""
    base_ts = parse_base_ts(base)

    ts = None
    if "time_local" in df.columns:
        # Try multiple formats; if fails, fall back to pandas parser
        # First pass: direct to_datetime coercion (handles ISO and many formats)
        ts = pd.to_datetime(
            df["time_local"], errors="coerce", infer_datetime_format=True
        )

        # Secondary targeted attempts if many NaT
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
        # Fallback: filename base
        if base_ts is not None:
            ts = pd.Series([base_ts] * len(df), index=df.index, dtype="datetime64[ns]")
        else:
            # Give up: no timestamps
            ts = pd.to_datetime(pd.Series([pd.NaT] * len(df)), errors="coerce")

    # If ms exists, add milliseconds
    if "ms" in df.columns:
        ms = pd.to_numeric(df["ms"], errors="coerce").fillna(0).astype("Int64")
        # Any missing ts stays NaT after addition
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
    # Drop non-positive frequencies for metric computations
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
    # Isolated if current is oob but neither neighbor is oob
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
    """
    Scan trailing comment lines '# key=value' and return a dict.
    If duplicates exist, the last wins.
    """
    out: Dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[::-1]:
            if not line.strip().startswith("#"):
                # stop when we reach non-comment tail
                break
            # Remove leading '#'
            txt = line.strip()[1:].strip()
            # Try key=value
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
        # Create a tiny placeholder image to avoid broken links
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
    # Keep order stable by sorting keys
    keys = list(d.keys())
    # Preferred ordering for time fields if present
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
    # Reorder keys so preferred come first
    keys_sorted = preferred + [k for k in keys if k not in preferred]
    seen = set()
    keys_sorted = [k for k in keys_sorted if not (k in seen or seen.add(k))]

    lines = ["| metric | value |", "|---|---|"]
    for k in keys_sorted:
        v = d.get(k, "")
        if isinstance(v, float) and math.isnan(v):
            s = ""
        elif isinstance(v, float):
            # fewer decimals for readability
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
    """
    Build a Markdown report string and optionally write it to out_md_path.
    """
    parts: List[str] = []
    parts.append(f"# PowerSentry Capture Report — {base}\n")

    parts.append("## Files")
    parts.append("")
    parts.append(f"- Level CSV: `{paths.get('level')}`")
    parts.append(f"- Frequency CSV: `{paths.get('freq')}`")
    parts.append("")

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

    # Interpretation blocks
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

    # Figures
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

    # Footers if any
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
    # Convert datetimes to strings
    d2 = {}
    for k, v in d.items():
        if isinstance(v, datetime):
            d2[k] = _fmt_dt(v)
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
) -> bool:
    """
    Analyze a pair (or a single file if one is missing). Returns True if at least one analysis succeeded.
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
        # Drop non-positive frequencies before any stats (per spec)
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

    # Report
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

    return have_any


def find_pairs_in_dir(dir_path: Path) -> List[SeriesFiles]:
    """
    Discover files in a directory and return a list of SeriesFiles with possible pairs.
    Level: <BASE>.csv
    Freq:  <BASE>F.csv
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return []

    # Map base -> level/freq paths
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


def main():
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
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()

    any_success = False

    if args.dir:
        pairs = find_pairs_in_dir(Path(args.dir).expanduser().resolve())
        if not pairs:
            print("No capture files found in directory.")
        for sf in pairs:
            ok = analyze_pair(
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
        # Base from either path
        base = None
        if level_path:
            base = parse_base_from_filename(level_path)
        if not base and freq_path:
            base = parse_base_from_filename(freq_path)
        if not base:
            # Fall back to stem sans 'F' if present
            base = (level_path.stem if level_path else freq_path.stem).rstrip("F")
        ok = analyze_pair(
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
        # Non-zero exit if nothing analyzable
        raise SystemExit(2)


if __name__ == "__main__":
    main()
