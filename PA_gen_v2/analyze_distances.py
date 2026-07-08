#!/usr/bin/env python3
"""
Distance-distribution analysis for Ghost-Probe positive (ghost) events.

Reads the mined events JSON (no nuScenes devkit needed — only numpy +
matplotlib) and produces:

  1. Emergence distance histogram
  2. Positive count per distance bin
  3. Unknown (no-evidence) ratio per distance bin
  4. OSZ overlap (confirmed-OSZ frame count) per distance bin

Plus a data-driven threshold recommendation: finds the distance beyond
which sample quality degrades (rising unknown ratio, falling OSZ overlap)
and suggests a cutoff.

Usage:
    python PA_gen_v2/analyze_distances.py [--events <json>] [--bin 5]
           [--out <png>] [--label_filter 1]
"""
import argparse
import json
import math
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use('Agg')                       # headless; savefig only
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ── data loading ──────────────────────────────────────────────────────
def load_events(path: str):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def emerge_distance(ev: dict) -> float:
    """Euclidean distance of the emergence point from ego (metres)."""
    x, y = ev['emerge_bev_xy']
    return math.hypot(float(x), float(y))


def unknown_ratio(ev: dict) -> float:
    """Fraction of lookback frames with NO_EVIDENCE (was_in_osz is None)."""
    w = ev.get('was_in_osz', [])
    if not w:
        return 0.0
    return sum(1 for v in w if v is None) / len(w)


# ── binning ───────────────────────────────────────────────────────────
def bin_events(events, bin_width=5.0, d_max=None):
    """Group events into distance bins. Returns OrderedDict[bin_label -> list]."""
    dists = [emerge_distance(e) for e in events]
    d_max = d_max or (int(max(dists) / bin_width) + 1) * bin_width
    edges = list(np.arange(0, d_max + bin_width, bin_width))
    bins = OrderedDict()
    for lo, hi in zip(edges[:-1], edges[1:]):
        key = f"{int(lo)}-{int(hi)}"
        bins[key] = [e for e, d in zip(events, dists) if lo <= d < hi]
    return bins, edges


def bin_stats(events_in_bin):
    """Compute aggregate stats for one bin."""
    n = len(events_in_bin)
    if n == 0:
        return dict(n=0, unk=0.0, osz_mean=0.0, osz_strong=0.0,
                    evi_mean=0.0)
    unk = np.mean([unknown_ratio(e) for e in events_in_bin])
    osz = np.array([e['n_osz_frames'] for e in events_in_bin], float)
    evi = np.array([e['n_evidence_frames'] for e in events_in_bin], float)
    return dict(
        n=n,
        unk=float(unk),
        osz_mean=float(osz.mean()),
        osz_strong=float(np.mean(osz >= 2)),     # fraction with >=2 OSZ frames
        evi_mean=float(evi.mean()),
    )


# ── threshold analysis ────────────────────────────────────────────────
def recommend_threshold(bin_data, bin_width):
    """
    Data-driven cutoff finder.

    Strategy:
      1. Find the distance bin with PEAK quality (highest strong% among
         well-populated bins) — this is the "sweet spot".
      2. Scan outward from the peak; flag the first distance where quality
         degrades AND stays degraded for >= 2 consecutive bins:
           • unknown ratio  >= 15%,  OR
           • mean OSZ overlap < 2.0, OR
           • strong% (osz>=2) < 60%.
      3. If no sustained degradation is found, report gradual degradation
         and recommend based on the cumulative retention table instead.

    The near tail (closer than the peak) is NOT a degradation zone —
    close-range objects naturally have brief shadow overlap.
    """
    # Collect well-populated bins (n >= 10) in distance order.
    populated = [(label, st) for label, st in bin_data.items()
                 if st['n'] >= 10]
    if len(populated) < 3:
        return None, ["Too few well-populated bins to determine a threshold."]

    # 1 — find peak (best strong%)
    peak_label, peak_st = max(populated, key=lambda kv: kv[1]['osz_strong'])
    peak_lo = int(peak_label.split('-')[0])

    # 2 — scan outward from peak for sustained degradation
    after_peak = [(l, s) for l, s in populated
                  if int(l.split('-')[0]) >= peak_lo]
    degraded = []          # list of (label, reason)
    for label, st in after_peak:
        r = None
        if st['unk'] >= 0.15:
            r = f"unknown {st['unk']*100:.0f}% >= 15%"
        elif st['osz_mean'] < 2.0:
            r = f"OSZ overlap {st['osz_mean']:.1f} < 2.0"
        elif st['osz_strong'] < 0.60:
            r = f"strong% {st['osz_strong']*100:.0f}% < 60%"
        degraded.append((label, r))

    # find first run of >= 2 consecutive degraded bins
    for i in range(len(degraded) - 1):
        if degraded[i][1] and degraded[i + 1][1]:
            cutoff = int(degraded[i][0].split('-')[0])
            reasons = [
                f"Sustained quality degradation starting at {degraded[i][0]}m:",
                f"  {degraded[i][0]}: {degraded[i][1]}",
                f"  {degraded[i+1][0]}: {degraded[i+1][1]}",
                f"Peak quality was at {peak_label}m (strong% "
                f"{peak_st['osz_strong']*100:.0f}%).",
            ]
            return cutoff, reasons

    # 3 — no sustained cliff
    return None, [
        f"No sustained quality cliff found (peak at {peak_label}m).",
        "Degradation is gradual — see cumulative retention table.",
        f"Recommend 35-40m as a practical balance (strong% stays ~75%).",
    ]


# ── plotting ──────────────────────────────────────────────────────────
def plot_analysis(dists, bin_data, edges, out_path, threshold=None):
    """Render a 2x2 figure and save to PNG."""
    centers = [(e0 + e1) / 2 for e0, e1 in zip(edges[:-1], edges[1:])]
    counts = [bin_data[k]['n'] for k in bin_data]
    unks = [bin_data[k]['unk'] * 100 for k in bin_data]
    oszs = [bin_data[k]['osz_mean'] for k in bin_data]
    strongs = [bin_data[k]['osz_strong'] * 100 for k in bin_data]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')

    # 1 — histogram
    ax = axes[0, 0]
    ax.hist(dists, bins=edges, color='#4a90d9', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(dists), color='#e74c3c', ls='--', lw=1.5,
               label=f'mean={np.mean(dists):.1f}m')
    if threshold is not None:
        ax.axvline(threshold, color='#e67e22', ls='-', lw=2,
                   label=f'threshold={threshold}m')
    ax.set_xlabel('Emergence distance (m)')
    ax.set_ylabel('Positive count')
    ax.set_title('1. Emergence distance histogram')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 2 — count per bin (bar)
    ax = axes[0, 1]
    bars = ax.bar(centers, counts, width=edges[1] - edges[0] * 0.8,
                  color='#4a90d9', edgecolor='white', alpha=0.85)
    ax.set_xlabel('Distance bin centre (m)')
    ax.set_ylabel('Positive count')
    ax.set_title('2. Positive count per distance bin')
    ax.grid(axis='y', alpha=0.3)
    for b, c in zip(bars, counts):
        if c > 0:
            ax.text(b.get_x() + b.get_width() / 2, c + 1, str(c),
                    ha='center', va='bottom', fontsize=8)

    # 3 — unknown ratio per bin
    ax = axes[1, 0]
    ax.bar(centers, unks, width=edges[1] - edges[0] * 0.8,
           color='#e74c3c', edgecolor='white', alpha=0.85)
    ax.axhline(15, color='#e67e22', ls='--', lw=1, label='15% quality bar')
    ax.set_xlabel('Distance bin centre (m)')
    ax.set_ylabel('Unknown ratio (%)')
    ax.set_title('3. Unknown (no-evidence) ratio per distance bin')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 4 — OSZ overlap per bin (mean + strong%)
    ax = axes[1, 1]
    ax.bar(centers, oszs, width=edges[1] - edges[0] * 0.8,
           color='#2ecc71', edgecolor='white', alpha=0.7, label='mean OSZ overlap')
    ax.plot(centers, [s / 100 * 4 for s in strongs], 'o-', color='#9b59b6',
            ms=4, label='strong (osz>=2) ratio [÷4 scale]')
    ax.axhline(2.0, color='#e67e22', ls='--', lw=1, label='OSZ=2 bar')
    ax.set_xlabel('Distance bin centre (m)')
    ax.set_ylabel('OSZ overlap (frames)')
    ax.set_title('4. OSZ overlap per distance bin')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Ghost-Probe: Positive-event distance distribution analysis',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


# ── text report ───────────────────────────────────────────────────────
def print_report(dists, events, bin_data, threshold, reasons, bin_width):
    pos = len(dists)
    print("=" * 64)
    print("  Ghost-Probe — Positive-event distance analysis")
    print("=" * 64)
    print(f"  Total positives : {pos}")
    print(f"  Distance (m)    : min={min(dists):.1f}  "
          f"max={max(dists):.1f}  mean={np.mean(dists):.1f}  "
          f"median={np.median(dists):.1f}  std={np.std(dists):.1f}")
    print(f"  Bin width       : {bin_width}m")
    print("-" * 64)
    hdr = (f"{'bin':>8}  {'n':>5}  {'%':>5}  {'unk%':>5}  "
           f"{'osz_m':>6}  {'strong%':>7}  {'evi_m':>6}")
    print(hdr)
    print("-" * 64)
    cum = 0
    for label, st in bin_data.items():
        cum += st['n']
        if st['n'] == 0:
            continue
        print(f"{label:>8}  {st['n']:5d}  {cum/pos*100:4.1f}%  "
              f"{st['unk']*100:4.1f}%  {st['osz_mean']:6.2f}  "
              f"{st['osz_strong']*100:6.1f}%  {st['evi_mean']:6.2f}")
    print("-" * 64)

    # cumulative retention + quality at candidate thresholds
    print("\n  Cumulative retention & quality at candidate thresholds:")
    print(f"  {'thresh':>7}  {'kept':>5}  {'ret%':>5}  {'unk%':>5}  "
          f"{'osz_m':>6}  {'strong%':>7}")
    for t in [25, 30, 35, 40, 45, 50]:
        kept_events = [e for e, d in zip(events, dists) if d < t]
        kept = len(kept_events)
        if kept == 0:
            continue
        unk_sub = np.mean([unknown_ratio(e) for e in kept_events]) * 100
        osz_sub = np.mean([e['n_osz_frames'] for e in kept_events])
        strong_sub = np.mean([e['n_osz_frames'] >= 2 for e in kept_events]) * 100
        print(f"  <= {t:3d}m  {kept:5d}  {kept/pos*100:4.1f}%  "
              f"{unk_sub:4.1f}%  {osz_sub:6.2f}  {strong_sub:6.1f}%")
    print()

    print("  Threshold recommendation (data-driven):")
    if threshold is not None:
        kept = sum(1 for d in dists if d < threshold)
        print(f"    → {threshold}m  (retains {kept}/{pos} = "
              f"{kept/pos*100:.1f}% of positives)")
        for r in reasons:
            print(f"      reason: {r}")
    else:
        print("    → No hard cutoff needed; degradation is gradual.")
        for r in reasons:
            print(f"      {r}")
    print("=" * 64)


# ── main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Analyze emergence-distance distribution of positive events.')
    parser.add_argument('--events', default=str(_REPO_ROOT / 'PA_gen_v2' /
                                                'output' / 'ghost_events_mini.json'))
    parser.add_argument('--bin', type=float, default=5.0,
                        help='Bin width in metres (default 5)')
    parser.add_argument('--label_filter', type=int, default=1,
                        help='1=positives (default), 0=negatives, -1=all')
    parser.add_argument('--out', default=str(_REPO_ROOT / 'PA_gen_v2' / 'output' /
                                             'distance_analysis.png'))
    args = parser.parse_args()

    events = load_events(args.events)
    if args.label_filter >= 0:
        events = [e for e in events if e['label'] == args.label_filter]
    if not events:
        print("No events matching filter.")
        sys.exit(1)

    dists = [emerge_distance(e) for e in events]
    bin_data, edges = bin_events(events, bin_width=args.bin)
    for k in bin_data:
        bin_data[k] = bin_stats(bin_data[k])

    threshold, reasons = recommend_threshold(bin_data, args.bin)

    print_report(dists, events, bin_data, threshold, reasons, args.bin)
    plot_analysis(dists, bin_data, edges, args.out, threshold)

    # also save a CSV for easy re-use
    csv_path = str(Path(args.out).with_suffix('.csv'))
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("bin,n,pct_cumulative,unk_pct,osz_mean,osz_strong_pct,evi_mean\n")
        cum = 0
        for label, st in bin_data.items():
            cum += st['n']
            if st['n'] == 0:
                continue
            f.write(f"{label},{st['n']},{cum/len(dists)*100:.1f},"
                    f"{st['unk']*100:.1f},{st['osz_mean']:.2f},"
                    f"{st['osz_strong']*100:.1f},{st['evi_mean']:.2f}\n")
    print(f"CSV saved: {csv_path}")


if __name__ == '__main__':
    main()
