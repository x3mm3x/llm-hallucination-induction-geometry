#!/usr/bin/env python3
"""
Generate publication figures from multi-run stability analysis.
================================================================

Reads multirun_aggregate.json (K=30) and produces:

  MAIN TEXT:
    fig1_forest.pdf/.png     — Effect-size stability forest plot
    fig2_pseudorep.pdf/.png  — Pseudoreplication demonstration

  APPENDIX:
    fig3_ctx_scatter.pdf/.png — Contextual norm–H scatter (representative run)
    fig4_pvalue_strips.pdf/.png — p-value distributions across K runs

Usage:
    python generate_figures.py

For fig3, requires raw_results_contextual.json and summary.json
from a single representative run. If absent, fig3 is skipped.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── Paths ──
AGGREGATE_PATH = './results_multirun/multirun_aggregate.json'
RAW_CTX_PATH = './results_multirun/raw_results_contextual.json'
CTX_SUMMARY_PATH = './results_multirun/summary_contextual.json'
OUTPUT_DIR = './figures_multirun'

# Fallback: also check current directory (for standalone runs)
_FALLBACK_RAW = './raw_results_contextual.json'
_FALLBACK_SUMMARY = './summary.json'
_FALLBACK_AGG = './multirun_aggregate.json'

# ── Style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

C_TYPE1 = '#d64541'
C_TYPE2 = '#2980b9'
C_TYPE3 = '#27ae60'


def get(obj, *keys, default=float('nan')):
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k, None)
        else:
            return default
        if obj is None:
            return default
    return obj


def _save(fig, name):
    for ext in ('.pdf', '.png'):
        fig.savefig(os.path.join(OUTPUT_DIR, name + ext))
    plt.close(fig)
    print(f'  ✓ {name}')


# ══════════════════════════════════════════════════════════════
#  FIGURE 1: FOREST PLOT
# ══════════════════════════════════════════════════════════════

def fig1_forest(data):
    pairs_order = ['type1_vs_type2', 'type1_vs_type3', 'type2_vs_type3']
    pair_labels = {'type1_vs_type2': 'T1 vs T2',
                   'type1_vs_type3': 'T1 vs T3',
                   'type2_vs_type3': 'T2 vs T3'}
    metrics_order = ['H', 'norm', 'max_sim']
    metric_labels = {'H': 'H(v)', 'norm': '‖v‖', 'max_sim': 'max sim'}

    fig, (ax_s, ax_c) = plt.subplots(
        1, 2, figsize=(7.2, 4.4), sharey=True,
        gridspec_kw={'wspace': 0.04, 'left': 0.22, 'right': 0.92})

    for ax, exp, title in [(ax_s, 'static', 'Static Embeddings'),
                            (ax_c, 'contextual', 'Contextual Hidden States')]:
        runs = data[exp]['runs']
        K = len(runs)
        y_pos = []
        y_labels = []
        y = 0
        group_boundaries = []

        for mi, m in enumerate(metrics_order):
            if mi > 0:
                group_boundaries.append(y - 0.3)
            for pair in pairs_order:
                rs = [get(r, 'two_level_stats', m, 'pairwise', pair,
                          'r_ci', 'r') for r in runs]
                rs = [x for x in rs if np.isfinite(x)]
                if not rs:
                    y += 1
                    continue

                med = np.median(rs)
                q25, q75 = np.percentile(rs, [25, 75])
                rmin, rmax = min(rs), max(rs)

                # Holm survival
                holm_ct = 0
                for r in runs:
                    hl = get(r, 'two_level_stats', m, 'holm', default=[])
                    for e in hl:
                        if isinstance(e, (list, tuple)) and pair in str(e[0]):
                            if e[3]:
                                holm_ct += 1
                            break
                holm_frac = holm_ct / K

                # Sig rate
                mw_ps = [get(r, 'two_level_stats', m, 'pairwise', pair,
                             'prompt_mean_mw', 'p') for r in runs]
                sig_rate = sum(1 for p in mw_ps
                               if np.isfinite(p) and p < 0.05) / K

                # Background by Holm rate
                if holm_frac >= 0.5:
                    ax.axhspan(y - 0.42, y + 0.42, color='#d5f4e6',
                               alpha=0.65, zorder=0, linewidth=0)
                elif holm_frac >= 0.15:
                    ax.axhspan(y - 0.42, y + 0.42, color='#fef9e7',
                               alpha=0.50, zorder=0, linewidth=0)

                # Whiskers
                ax.plot([rmin, rmax], [y, y], color='#b0b8c0',
                        linewidth=0.7, solid_capstyle='round', zorder=1)
                ax.plot([q25, q75], [y, y], color='#34495e',
                        linewidth=3.2, solid_capstyle='round', zorder=2)
                ax.plot(med, y, 'o', color='#1a252f', markersize=4.5,
                        zorder=3, markeredgecolor='white',
                        markeredgewidth=0.4)

                # Sig annotation on outer edge
                sig_n = int(sig_rate * K)
                if ax == ax_c:
                    ax.text(1.12, y, f'{sig_n}/{K}',
                            fontsize=6, ha='left', va='center',
                            color='#666666', family='monospace')

                y_pos.append(y)
                if ax == ax_s:
                    y_labels.append(
                        f'{metric_labels[m]}   {pair_labels[pair]}')
                y += 1
            y += 0.6

        # Zero line
        ax.axvline(0, color='#cc3333', linewidth=0.7, linestyle='-',
                    alpha=0.4, zorder=0)

        ax.set_xlim(-1.15, 1.15)
        ax.set_xlabel('Rank-biserial $r$', fontsize=9)
        ax.set_title(title, fontweight='bold', pad=8, fontsize=10.5)
        ax.grid(axis='x', alpha=0.12, linewidth=0.4)

        for gb in group_boundaries:
            ax.axhline(gb, color='#d0d0d0', linewidth=0.4)

    ax_s.set_yticks(y_pos)
    ax_s.set_yticklabels(y_labels, fontsize=8.5)
    ax_s.invert_yaxis()

    # Sig/K header
    ax_c.text(1.12, -0.9, 'sig/20', fontsize=6, ha='left',
              va='center', color='#999999', family='monospace',
              style='italic')

    # Legend
    legend_elements = [
        Line2D([0], [0], color='#34495e', linewidth=3.2, label='IQR'),
        Line2D([0], [0], color='#b0b8c0', linewidth=0.7,
               label='Full range'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#1a252f', markersize=4.5,
               label='Median $r$'),
        Patch(facecolor='#d5f4e6', edgecolor='#bbb', linewidth=0.4,
              label='Holm ≥ 50%'),
        Patch(facecolor='#fef9e7', edgecolor='#bbb', linewidth=0.4,
              label='Holm ≥ 15%'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=5, frameon=False, fontsize=7,
               bbox_to_anchor=(0.55, -0.04))

    _save(fig, 'fig1_forest')


# ══════════════════════════════════════════════════════════════
#  FIGURE 2: PSEUDOREPLICATION
# ══════════════════════════════════════════════════════════════

def fig2_pseudorep(data):
    runs = data['contextual']['runs']
    K = len(runs)
    metrics = ['H', 'norm', 'max_sim']
    pairs = ['type1_vs_type2', 'type1_vs_type3', 'type2_vs_type3']
    pair_labels = {'type1_vs_type2': 'T1–T2',
                   'type1_vs_type3': 'T1–T3',
                   'type2_vs_type3': 'T2–T3'}
    metric_labels = {'H': 'H(v)', 'norm': '‖h‖', 'max_sim': 'max sim'}

    rows = []
    for m in metrics:
        for pair in pairs:
            tok_ps = [get(r, 'two_level_stats', m, 'pairwise', pair,
                          'token', 'p') for r in runs]
            pr_ps = [get(r, 'two_level_stats', m, 'pairwise', pair,
                         'prompt_mean_mw', 'p') for r in runs]
            tok_sig = sum(1 for p in tok_ps
                          if np.isfinite(p) and p < 0.05) / K
            pr_sig = sum(1 for p in pr_ps
                         if np.isfinite(p) and p < 0.05) / K
            label = f'{metric_labels[m]}  {pair_labels[pair]}'
            rows.append((label, tok_sig, pr_sig))

    rows.sort(key=lambda x: -(x[1] - x[2]))

    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    for i, (label, tok, pr) in enumerate(rows):
        y = len(rows) - 1 - i

        # Connector
        ax.plot([pr, tok], [y, y], color='#d5d8dc', linewidth=2.0,
                zorder=1, solid_capstyle='round')

        # Prompt (square, dark)
        ax.plot(pr, y, 's', color='#2c3e50', markersize=6.5, zorder=3,
                markeredgecolor='white', markeredgewidth=0.4)

        # Token (circle, red)
        ax.plot(tok, y, 'o', color='#e74c3c', markersize=6.5, zorder=3,
                markeredgecolor='white', markeredgewidth=0.4)

        # Inflation gap annotation
        gap = tok - pr
        if gap > 0.08:
            ax.annotate(f'{gap:.0%}',
                        xy=((tok + pr) / 2, y), fontsize=5.5,
                        ha='center', va='bottom', color='#999',
                        xytext=(0, 3), textcoords='offset points')

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel('Significance rate across 20 runs', fontsize=9)
    ax.set_title('Contextual: Token vs Prompt Significance',
                 fontweight='bold', pad=8, fontsize=10)

    ax.grid(axis='x', alpha=0.12, linewidth=0.4)

    legend_elements = [
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor='#2c3e50', markersize=6.5,
               label='Prompt-level ($N$=15)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#e74c3c', markersize=6.5,
               label='Token-level ($N$≈900)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              frameon=True, framealpha=0.95, edgecolor='#ddd',
              fontsize=7)

    _save(fig, 'fig2_pseudorep')


# ══════════════════════════════════════════════════════════════
#  FIGURE 3: CONTEXTUAL SCATTER
# ══════════════════════════════════════════════════════════════

def fig3_ctx_scatter(summary_path, raw_ctx_path):
    with open(raw_ctx_path) as f:
        raw = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)

    zones = summary.get('zones', {})

    type_colors = {'type1': C_TYPE1, 'type2': C_TYPE2, 'type3': C_TYPE3}
    type_labels = {'type1': 'Type 1\n(center-drift)',
                   'type2': 'Type 2\n(wrong-well)',
                   'type3': 'Type 3\n(coverage gap)'}

    by_type = {t: {'norm': [], 'H': []} for t in ['type1', 'type2', 'type3']}
    for seq in raw:
        for m in seq.get('measurements', []):
            by_type[seq['type']]['norm'].append(m['norm'])
            by_type[seq['type']]['H'].append(m['H'])

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True,
                              gridspec_kw={'wspace': 0.05})

    for ax, t in zip(axes, ['type1', 'type2', 'type3']):
        norms = by_type[t]['norm']
        Hs = by_type[t]['H']
        n = len(norms)

        ax.scatter(norms, Hs, c=type_colors[t], s=5, alpha=0.40,
                   edgecolors='none', rasterized=True)

        # Zone thresholds
        if 'H_low' in zones:
            ax.axhline(zones['H_low'], color='#bbb', linewidth=0.5,
                       linestyle='--', alpha=0.6)
        if 'H_high' in zones:
            ax.axhline(zones['H_high'], color='#bbb', linewidth=0.5,
                       linestyle='--', alpha=0.6)

        ax.set_xlabel('‖h‖', fontsize=9)
        ax.set_title(type_labels[t], fontsize=8.5, fontweight='bold',
                     color=type_colors[t], pad=4, linespacing=1.1)

        ax.text(0.97, 0.03, f'$n$={n}', transform=ax.transAxes,
                fontsize=6.5, ha='right', va='bottom', color='#999')
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel('H(v)', fontsize=9)

    fig.suptitle('Contextual Hidden-State Signatures',
                 fontsize=10, fontweight='bold', y=1.06)
    fig.text(0.5, 0.99, '(representative run)',
             ha='center', fontsize=7.5, color='#888', style='italic')

    _save(fig, 'fig3_ctx_scatter')


# ══════════════════════════════════════════════════════════════
#  FIGURE 4: P-VALUE STRIPS
# ══════════════════════════════════════════════════════════════

def fig4_pvalue_strips(data):
    pairs = ['type1_vs_type2', 'type1_vs_type3', 'type2_vs_type3']
    pair_labels = ['T1–T2', 'T1–T3', 'T2–T3']
    pair_colors = ['#7f8c8d', C_TYPE1, C_TYPE2]

    fig, (ax_s, ax_c) = plt.subplots(
        1, 2, figsize=(6.5, 3.0), sharey=True,
        gridspec_kw={'wspace': 0.06})

    rng = np.random.RandomState(42)

    for ax, exp, title in [(ax_s, 'static', 'Static — Norm'),
                            (ax_c, 'contextual', 'Contextual — Norm')]:
        runs = data[exp]['runs']
        K = len(runs)

        for i, (pair, pc, pl) in enumerate(
                zip(pairs, pair_colors, pair_labels)):
            ps = [get(r, 'two_level_stats', 'norm', 'pairwise',
                       pair, 'prompt_mean_mw', 'p') for r in runs]
            ps = [p for p in ps if np.isfinite(p)]

            jitter = rng.uniform(-0.15, 0.15, len(ps))
            ax.scatter([i + j for j in jitter], ps, c=pc, s=18,
                       alpha=0.65, edgecolors='white', linewidth=0.3,
                       zorder=3)

            med = np.median(ps)
            ax.plot([i - 0.25, i + 0.25], [med, med], color=pc,
                    linewidth=2.2, zorder=4, solid_capstyle='round')

            sig = sum(1 for p in ps if p < 0.05)
            ax.text(i, 1.08, f'{sig}/{K}', ha='center', fontsize=6.5,
                    color=pc, fontweight='bold')

        ax.axhline(0.05, color='#cc3333', linewidth=0.7,
                    linestyle='--', alpha=0.5, zorder=1)
        ax.text(2.55, 0.058, 'α', fontsize=7, color='#cc3333',
                va='bottom', style='italic')

        ax.set_xticks(range(3))
        ax.set_xticklabels(pair_labels)
        ax.set_title(title, fontweight='bold', pad=10, fontsize=10)
        ax.set_ylim(-0.03, 1.15)
        ax.grid(axis='y', alpha=0.10, linewidth=0.4)

    ax_s.set_ylabel('Prompt-level MW $p$', fontsize=9)

    _save(fig, 'fig4_pvalue_strips')


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Resolve aggregate path (with fallback)
    agg_path = AGGREGATE_PATH
    if not os.path.exists(agg_path):
        agg_path = _FALLBACK_AGG
    if not os.path.exists(agg_path):
        print(f'ERROR: Cannot find multirun_aggregate.json')
        print(f'  Checked: {AGGREGATE_PATH}')
        print(f'  Checked: {_FALLBACK_AGG}')
        return

    print(f'Loading {agg_path}...')
    with open(agg_path) as f:
        data = json.load(f)
    print(f'  K = {data["K"]}\n')

    print('Generating figures:')
    fig1_forest(data)
    fig2_pseudorep(data)
    fig4_pvalue_strips(data)

    # Resolve scatter data paths (with fallback)
    raw_path = RAW_CTX_PATH if os.path.exists(RAW_CTX_PATH) else _FALLBACK_RAW
    sum_path = CTX_SUMMARY_PATH if os.path.exists(CTX_SUMMARY_PATH) else _FALLBACK_SUMMARY

    if os.path.exists(raw_path) and os.path.exists(sum_path):
        fig3_ctx_scatter(sum_path, raw_path)
    else:
        print(f'  ⊘ fig3: need raw_results_contextual.json + summary.json')
        print(f'    Checked: {RAW_CTX_PATH}, {_FALLBACK_RAW}')
        print(f'    Checked: {CTX_SUMMARY_PATH}, {_FALLBACK_SUMMARY}')
        print(f'    Re-run run_multirun.py (it saves these automatically)')

    print(f'\nAll figures in {os.path.abspath(OUTPUT_DIR)}/')


if __name__ == '__main__':
    main()
