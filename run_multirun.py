#!/usr/bin/env python3
"""
Multi-Run Stability Analysis for Hallucination Induction
=========================================================

Runs both the static and contextual hallucination induction pipelines
K times with different generation seeds, then aggregates results to
quantify which findings are stable across runs vs. seed-dependent.

Architecture
------------
  - Calibration (vocabulary filtering, clustering, zone thresholds)
    is performed ONCE per experiment with a fixed seed (42).
  - Text generation is repeated K times with seeds 1..K.
  - Statistical tests use a fixed internal seed (42 in
    hallucination_stats.py), so analysis is deterministic given
    the same generated data.

This isolates generation stochasticity as the sole source of
run-to-run variation.

Output (in ./results_multirun/)
-------------------------------
  multirun_aggregate.json  — full per-run + aggregate statistics
  multirun_report.txt      — human-readable stability report

Usage
-----
  python run_multirun.py

Runtime estimate: ~(10 + 25) min × K
  K=10  → ~6 h   |   K=20  → ~12 h

Prerequisites
-------------
  hallucination_induction.py, hallucination_induction_contextual.py,
  hallucination_stats.py must be in the same directory.
"""

# ═══════════════════════════════════════════════════════════════
#  CHANGE THESE VALUES
# ═══════════════════════════════════════════════════════════════
K = 20
REPRESENTATIVE_SEED = 7   # Save raw per-token data for this seed (for figures)
# ═══════════════════════════════════════════════════════════════

import os
import sys
import time
import json
import gc
import warnings
import numpy as np

warnings.filterwarnings('ignore')

from hallucination_stats import results_to_json

OUTPUT_DIR = './results_multirun'


def _save_representative(all_results, zones, experiment_name):
    """Save raw per-token results and zones from one representative run.

    These files are needed by generate_figures.py for the scatter plot.
    """
    raw_path = os.path.join(
        OUTPUT_DIR, f'raw_results_{experiment_name}.json')
    summary_path = os.path.join(
        OUTPUT_DIR, f'summary_{experiment_name}.json')

    # Serialise raw results (strip numpy types)
    raw = []
    for seq in all_results:
        entry = {
            'type': seq['type'],
            'prompt': seq['prompt'],
            'measurements': [],
        }
        for m in seq.get('measurements', []):
            entry['measurements'].append({
                k: (float(v) if hasattr(v, 'item') else v)
                for k, v in m.items()
            })
        raw.append(entry)

    with open(raw_path, 'w') as f:
        json.dump(raw, f)

    # Serialise zones (threshold dict)
    zones_ser = {k: float(v) if hasattr(v, 'item') else v
                 for k, v in zones.items()}
    with open(summary_path, 'w') as f:
        json.dump({'zones': zones_ser}, f, indent=2)

    print(f"  ✓ Saved representative raw data: {raw_path}")
    print(f"  ✓ Saved representative zones: {summary_path}")


# ──────────────────────────────────────────────────────────────
# STATIC EXPERIMENT — K RUNS
# ──────────────────────────────────────────────────────────────

def run_static_multirun(K, static_mod):
    """Run the static pipeline K times with fixed calibration.

    Calibration (vocabulary filtering, clustering, zone thresholds)
    is deterministic — it depends only on GPT-2's embedding matrix
    and the numpy seed.  Only generation varies across runs.
    """
    import torch
    print("=" * 70)
    print(f"STATIC EXPERIMENT — {K} RUNS")
    print("=" * 70)
    t_exp = time.time()

    # ── Calibration (once) ──
    np.random.seed(42)
    model, tokenizer, emb_matrix = static_mod.load_gpt2()
    filtered_emb, filtered_indices, words, self_info = \
        static_mod.filter_vocabulary(emb_matrix, tokenizer)
    kmeans, centroids, zones, vocab_stats = \
        static_mod.cluster_and_calibrate(filtered_emb, words, self_info)

    # ── K generation runs ──
    runs = []
    for k in range(K):
        seed = k + 1
        print(f"\n{'─' * 60}")
        print(f"  STATIC run {k + 1}/{K}  (seed={seed})")
        print(f"{'─' * 60}")
        t0 = time.time()

        torch.manual_seed(seed)

        gen1 = static_mod.generate_sequences(
            model, tokenizer, static_mod.TYPE1_PROMPTS, 'type1')
        gen2 = static_mod.generate_sequences(
            model, tokenizer, static_mod.TYPE2_PROMPTS, 'type2')
        gen3 = static_mod.generate_sequences(
            model, tokenizer, static_mod.TYPE3_PROMPTS, 'type3')
        all_results = gen1 + gen2 + gen3

        all_results = static_mod.measure_geometry(
            all_results, emb_matrix, centroids, zones)
        confusion = static_mod.compute_confusion_matrix(all_results)
        stat_tests, diagnostics = static_mod.run_statistical_tests(all_results)

        run_data = {
            'seed': seed,
            'two_level_stats': results_to_json(stat_tests),
            'confusion': {
                'fractions': confusion['fractions'],
                'diagonal': [float(d) for d in confusion['diagonal']],
                'mean_diagonal': float(confusion['mean_diagonal']),
                'totals': {t: int(confusion['totals'][t])
                           for t in confusion['totals']},
            },
            'runtime_s': round(time.time() - t0, 1),
        }
        runs.append(run_data)

        del all_results, gen1, gen2, gen3, confusion, stat_tests, diagnostics
        gc.collect()

        print(f"  ✓ Static run {k + 1} done in {run_data['runtime_s']:.0f}s")

    del model
    gc.collect()

    print(f"\n  Static experiment total: {time.time() - t_exp:.0f}s")
    return runs


# ──────────────────────────────────────────────────────────────
# CONTEXTUAL EXPERIMENT — K RUNS
# ──────────────────────────────────────────────────────────────

def run_contextual_multirun(K, ctx_mod):
    """Run the contextual pipeline K times with fixed calibration.

    Calibration (background generation + clustering) is run once
    under seed 42.  This isolates experimental generation variance
    from calibration variance — a fix over the standalone script
    where both shared one seed.
    """
    import torch
    print("\n" + "=" * 70)
    print(f"CONTEXTUAL EXPERIMENT — {K} RUNS")
    print("=" * 70)
    t_exp = time.time()

    # ── Load model ──
    np.random.seed(42)
    model, tokenizer, static_emb = ctx_mod.load_gpt2()

    # ── Calibration (once, fixed seed) ──
    torch.manual_seed(42)
    calibration_hidden = ctx_mod.build_calibration_distribution(
        model, tokenizer)
    kmeans, centroids, zones, calib_stats = \
        ctx_mod.cluster_and_calibrate(calibration_hidden)
    del calibration_hidden
    gc.collect()

    # ── K generation runs ──
    runs = []
    for k in range(K):
        seed = k + 1
        print(f"\n{'─' * 60}")
        print(f"  CONTEXTUAL run {k + 1}/{K}  (seed={seed})")
        print(f"{'─' * 60}")
        t0 = time.time()

        torch.manual_seed(seed)

        gen1 = ctx_mod.generate_experimental(
            model, tokenizer, ctx_mod.TYPE1_PROMPTS, 'type1')
        gen2 = ctx_mod.generate_experimental(
            model, tokenizer, ctx_mod.TYPE2_PROMPTS, 'type2')
        gen3 = ctx_mod.generate_experimental(
            model, tokenizer, ctx_mod.TYPE3_PROMPTS, 'type3')
        all_results = gen1 + gen2 + gen3

        all_results = ctx_mod.measure_geometry(
            all_results, centroids, zones, static_emb=static_emb)
        confusion = ctx_mod.compute_confusion_matrix(all_results)
        stat_tests, diagnostics = ctx_mod.run_statistical_tests(all_results)

        run_data = {
            'seed': seed,
            'two_level_stats': results_to_json(stat_tests),
            'confusion': {
                'fractions': confusion['fractions'],
                'diagonal': [float(d) for d in confusion['diagonal']],
                'mean_diagonal': float(confusion['mean_diagonal']),
                'totals': {t: int(confusion['totals'][t])
                           for t in confusion['totals']},
            },
            'runtime_s': round(time.time() - t0, 1),
        }
        runs.append(run_data)

        # Save raw per-token data for representative run (for scatter fig)
        if seed == REPRESENTATIVE_SEED:
            _save_representative(all_results, zones, 'contextual')

        del all_results, gen1, gen2, gen3, confusion, stat_tests, diagnostics
        gc.collect()

        print(f"  ✓ Contextual run {k + 1} done in {run_data['runtime_s']:.0f}s")

    del model
    gc.collect()

    print(f"\n  Contextual experiment total: {time.time() - t_exp:.0f}s")
    return runs


# ──────────────────────────────────────────────────────────────
# AGGREGATION
# ──────────────────────────────────────────────────────────────

def _get(d, *keys, default=float('nan')):
    """Safely drill into nested dicts."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, None)
        else:
            return default
        if d is None:
            return default
    return d


def _summarise(values):
    """Compute summary statistics for a list of floats."""
    v = np.array([x for x in values if np.isfinite(x)])
    if len(v) == 0:
        return {'n': 0, 'median': float('nan'), 'mean': float('nan'),
                'std': float('nan'), 'iqr_lo': float('nan'),
                'iqr_hi': float('nan'), 'prop_sig': float('nan'),
                'values': []}
    return {
        'n': int(len(v)),
        'median': float(np.median(v)),
        'mean': float(np.mean(v)),
        'std': float(np.std(v)),
        'iqr_lo': float(np.percentile(v, 25)),
        'iqr_hi': float(np.percentile(v, 75)),
        'values': [float(x) for x in v],
    }


def _prop_sig(values, alpha=0.05):
    """Fraction of values below alpha."""
    v = [x for x in values if np.isfinite(x)]
    return float(np.mean([p < alpha for p in v])) if v else float('nan')


def aggregate_runs(runs, experiment_name):
    """Aggregate K runs into stability statistics.

    Returns a dict with:
      - per-metric omnibus p-value distributions
      - per-pair prompt-level p, effect size, Holm survival rates
      - confusion matrix mean/std
      - per-run data preserved for reference
    """
    K = len(runs)
    metrics = ['H', 'norm', 'max_sim']
    pairs = ['type1_vs_type2', 'type1_vs_type3', 'type2_vs_type3']
    conditions = ['type1', 'type2', 'type3']

    agg = {
        'experiment': experiment_name,
        'K': K,
        'seeds': [r['seed'] for r in runs],
        'runtimes_s': [r['runtime_s'] for r in runs],
    }

    # ── Per-metric aggregation ──
    for m in metrics:
        m_agg = {}

        # Omnibus KW p-values
        for level, path in [
            ('token_kw',           ('token_kw', 'p')),
            ('prompt_kw',          ('prompt_mean_kw', 'p')),
            ('prompt_kw_perm',     ('prompt_mean_kw_perm', 'p')),
            ('prompt_kw_median',   ('prompt_median_kw', 'p')),
        ]:
            vals = [_get(r, 'two_level_stats', m, *path) for r in runs]
            s = _summarise(vals)
            s['prop_sig'] = _prop_sig(vals)
            m_agg[level] = s

        # Pairwise tests
        for pair in pairs:
            pair_agg = {}

            # Prompt-level MW p
            vals = [_get(r, 'two_level_stats', m, 'pairwise',
                         pair, 'prompt_mean_mw', 'p') for r in runs]
            s = _summarise(vals)
            s['prop_sig'] = _prop_sig(vals)
            pair_agg['prompt_mw_p'] = s

            # Permutation p
            vals = [_get(r, 'two_level_stats', m, 'pairwise',
                         pair, 'prompt_mean_perm', 'p') for r in runs]
            s = _summarise(vals)
            s['prop_sig'] = _prop_sig(vals)
            pair_agg['perm_p'] = s

            # Token-level MW p (for pseudoreplication comparison)
            vals = [_get(r, 'two_level_stats', m, 'pairwise',
                         pair, 'token', 'p') for r in runs]
            s = _summarise(vals)
            s['prop_sig'] = _prop_sig(vals)
            pair_agg['token_p'] = s

            # Rank-biserial r
            vals = [_get(r, 'two_level_stats', m, 'pairwise',
                         pair, 'r_ci', 'r') for r in runs]
            pair_agg['r'] = _summarise(vals)

            # Diff CI
            vals_lo = [_get(r, 'two_level_stats', m, 'pairwise',
                            pair, 'diff_ci', 'lo') for r in runs]
            vals_hi = [_get(r, 'two_level_stats', m, 'pairwise',
                            pair, 'diff_ci', 'hi') for r in runs]
            pair_agg['diff_ci_lo'] = _summarise(vals_lo)
            pair_agg['diff_ci_hi'] = _summarise(vals_hi)

            # Holm survival rate
            holm_survived = []
            for r in runs:
                holm_list = _get(r, 'two_level_stats', m, 'holm',
                                 default=[])
                survived = False
                for entry in holm_list:
                    if isinstance(entry, (list, tuple)) and pair in str(entry[0]):
                        survived = bool(entry[3])
                        break
                holm_survived.append(survived)
            pair_agg['holm_prop_sig'] = float(np.mean(holm_survived))
            pair_agg['holm_values'] = holm_survived

            m_agg[pair] = pair_agg

        # Condition means
        for cond in conditions:
            vals = [_get(r, 'two_level_stats', m, 'condition_stats',
                         cond, 'prompt_mean') for r in runs]
            m_agg[f'{cond}_prompt_mean'] = _summarise(vals)

        agg[m] = m_agg

    # ── Confusion matrix aggregation ──
    cm_agg = {}
    for typ in conditions:
        for zone in conditions + ['unclassified']:
            vals = [_get(r, 'confusion', 'fractions', typ, zone)
                    for r in runs]
            cm_agg[f'{typ}->{zone}'] = _summarise(vals)

    diags = [_get(r, 'confusion', 'mean_diagonal') for r in runs]
    cm_agg['mean_diagonal'] = _summarise(diags)
    agg['confusion'] = cm_agg

    # ── Pseudoreplication diagnostic ──
    # For each metric, compare token-level and prompt-level significance rates
    pseudo = {}
    for m in metrics:
        tok_rates = []
        pr_rates = []
        for pair in pairs:
            tok_ps = [_get(r, 'two_level_stats', m, 'pairwise',
                           pair, 'token', 'p') for r in runs]
            pr_ps = [_get(r, 'two_level_stats', m, 'pairwise',
                          pair, 'prompt_mean_mw', 'p') for r in runs]
            tok_rates.append(_prop_sig(tok_ps))
            pr_rates.append(_prop_sig(pr_ps))
        pseudo[m] = {
            'token_sig_rate_per_pair': tok_rates,
            'prompt_sig_rate_per_pair': pr_rates,
            'pairs': pairs,
        }
    agg['pseudoreplication'] = pseudo

    return agg


# ──────────────────────────────────────────────────────────────
# REPORT FORMATTING
# ──────────────────────────────────────────────────────────────

def format_report(static_agg, ctx_agg, K):
    """Generate human-readable stability report."""
    lines = []
    w = 76

    lines.append("=" * w)
    lines.append("MULTI-RUN STABILITY ANALYSIS")
    lines.append(f"K = {K} independent generation runs per experiment")
    lines.append(f"Calibration: fixed (seed=42) | Generation seeds: 1..{K}")
    lines.append("=" * w)

    for agg, label in [(static_agg, 'STATIC'), (ctx_agg, 'CONTEXTUAL')]:
        lines.append(f"\n{'=' * w}")
        lines.append(f"  {label} EXPERIMENT")
        lines.append(f"{'=' * w}")

        rts = agg.get('runtimes_s', [])
        if rts:
            lines.append(f"  Runtime per run: "
                         f"mean={np.mean(rts):.0f}s, "
                         f"total={np.sum(rts):.0f}s")

        # ── Omnibus ──
        lines.append(f"\n  OMNIBUS KRUSKAL-WALLIS (prompt-level, N=15/group)")
        lines.append(f"  {'Metric':<10} {'Median p':>10} "
                     f"{'IQR':>16} {'Sig rate':>10} "
                     f"{'Perm med':>10} {'Perm sig':>10}")
        lines.append(f"  {'-' * 68}")

        for m in ['H', 'norm', 'max_sim']:
            pkw = agg[m]['prompt_kw']
            ppkw = agg[m]['prompt_kw_perm']
            lines.append(
                f"  {m:<10} {pkw['median']:>10.4f} "
                f"[{pkw['iqr_lo']:.3f}, {pkw['iqr_hi']:.3f}] "
                f"{pkw['prop_sig']:>8.0%}   "
                f"{ppkw['median']:>10.4f} {ppkw['prop_sig']:>8.0%}")

        # ── Pairwise ──
        lines.append(f"\n  PAIRWISE (prompt-level Mann-Whitney)")
        lines.append(
            f"  {'Metric':<8} {'Pair':<16} {'Med p':>8} "
            f"{'Sig':>6} {'Holm':>6} {'Med r':>8} "
            f"{'r SD':>8} {'Perm sig':>10}")
        lines.append(f"  {'-' * 74}")

        pairs = ['type1_vs_type2', 'type1_vs_type3', 'type2_vs_type3']
        pair_short = {'type1_vs_type2': 'T1-T2',
                      'type1_vs_type3': 'T1-T3',
                      'type2_vs_type3': 'T2-T3'}

        for m in ['H', 'norm', 'max_sim']:
            for pair in pairs:
                pa = agg[m][pair]
                mw = pa['prompt_mw_p']
                pm = pa['perm_p']
                r = pa['r']
                holm = pa['holm_prop_sig']
                lines.append(
                    f"  {m:<8} {pair_short[pair]:<16} "
                    f"{mw['median']:>8.4f} "
                    f"{mw['prop_sig']:>5.0%} "
                    f"{holm:>5.0%} "
                    f"{r['median']:>+8.3f} "
                    f"{r['std']:>8.3f} "
                    f"{pm['prop_sig']:>8.0%}")
            lines.append("")

        # ── Condition means ──
        lines.append(f"  CONDITION MEANS (prompt-level, across runs)")
        lines.append(f"  {'Metric':<10} {'Type 1':>18} "
                     f"{'Type 2':>18} {'Type 3':>18}")
        lines.append(f"  {'-' * 66}")
        for m in ['H', 'norm', 'max_sim']:
            vals = []
            for cond in ['type1', 'type2', 'type3']:
                s = agg[m][f'{cond}_prompt_mean']
                vals.append(f"{s['mean']:.4f} +/- {s['std']:.4f}")
            lines.append(f"  {m:<10} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")

        # ── Confusion ──
        lines.append(f"\n  CONFUSION MATRIX (mean +/- std across {K} runs)")
        zones = ['type1', 'type2', 'type3', 'unclassified']
        lines.append(f"  {'':>10} {'Z1':>14} {'Z2':>14} "
                     f"{'Z3':>14} {'Unc':>14}")
        lines.append(f"  {'-' * 68}")
        for typ in ['type1', 'type2', 'type3']:
            cells = []
            for zone in zones:
                s = agg['confusion'][f'{typ}->{zone}']
                cells.append(f"{s['mean']:.3f}+/-{s['std']:.3f}")
            lines.append(f"  {typ:>10} {cells[0]:>14} {cells[1]:>14} "
                         f"{cells[2]:>14} {cells[3]:>14}")

        md = agg['confusion']['mean_diagonal']
        lines.append(f"  Mean diagonal: {md['mean']:.3f} +/- {md['std']:.3f}")

        # ── Pseudoreplication diagnostic ──
        lines.append(f"\n  PSEUDOREPLICATION DIAGNOSTIC")
        lines.append(f"  (fraction of runs where comparison is significant)")
        lines.append(f"  {'Metric':<10} {'Pair':<16} "
                     f"{'Token sig':>12} {'Prompt sig':>12} {'Ratio':>8}")
        lines.append(f"  {'-' * 60}")
        for m in ['H', 'norm', 'max_sim']:
            ps = agg['pseudoreplication'][m]
            for i, pair in enumerate(ps['pairs']):
                tok = ps['token_sig_rate_per_pair'][i]
                pr = ps['prompt_sig_rate_per_pair'][i]
                ratio = f"{tok/pr:.1f}x" if pr > 0 else ("—" if tok == 0 else "inf")
                lines.append(
                    f"  {m:<10} {pair_short[pair]:<16} "
                    f"{tok:>10.0%}   {pr:>10.0%}   {ratio:>8}")
            lines.append("")

    # ── Cross-experiment summary ──
    lines.append("=" * w)
    lines.append("  CROSS-EXPERIMENT STABILITY SUMMARY")
    lines.append("=" * w)

    lines.append(f"\n  Key claims and their stability across {K} runs:\n")

    for agg, label in [(static_agg, 'Static'), (ctx_agg, 'Contextual')]:
        lines.append(f"  {label}:")

        # Type 3 norm separation
        for pair, short in [('type1_vs_type3', 'T1-T3'),
                            ('type2_vs_type3', 'T2-T3')]:
            pa = agg['norm'][pair]
            lines.append(
                f"    Norm {short}: sig {pa['prompt_mw_p']['prop_sig']:.0%}, "
                f"Holm {pa['holm_prop_sig']:.0%}, "
                f"median r={pa['r']['median']:+.3f}")

        # Type 1/2 non-separation
        pa12 = agg['norm']['type1_vs_type2']
        lines.append(
            f"    Norm T1-T2: sig {pa12['prompt_mw_p']['prop_sig']:.0%} "
            f"(non-separation {'confirmed' if pa12['prompt_mw_p']['prop_sig'] <= 0.1 else 'NOT confirmed'})")

        # H non-significance at prompt level
        h_omnibus = agg['H']['prompt_kw']
        lines.append(
            f"    H omnibus: sig {h_omnibus['prop_sig']:.0%} "
            f"(pseudoreplication artifact {'confirmed' if h_omnibus['prop_sig'] <= 0.1 else 'NOT confirmed'})")

        lines.append("")

    lines.append("=" * w)
    lines.append("END OF MULTI-RUN REPORT")
    lines.append("=" * w)

    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"MULTI-RUN STABILITY ANALYSIS  (K = {K})")
    print(f"Static + Contextual × {K} generation runs each")
    est_min = K * 35
    print(f"Estimated runtime: ~{est_min // 60}h {est_min % 60}m")
    print("=" * 70)

    t_total = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Import pipeline modules (deferred to avoid torch import at parse time)
    import hallucination_induction as static_mod
    import hallucination_induction_contextual as ctx_mod

    # Disable figure generation in both modules
    static_mod.CONFIG['generate_figures'] = False
    ctx_mod.CONFIG['generate_figures'] = False

    # ── Run experiments ──
    static_runs = run_static_multirun(K, static_mod)
    ctx_runs = run_contextual_multirun(K, ctx_mod)

    # ── Aggregate ──
    print(f"\n{'=' * 70}")
    print("AGGREGATING RESULTS")
    print(f"{'=' * 70}")

    static_agg = aggregate_runs(static_runs, 'static')
    ctx_agg = aggregate_runs(ctx_runs, 'contextual')

    # ── Save JSON ──
    output = {
        'K': K,
        'total_runtime_s': round(time.time() - t_total, 1),
        'static': {
            'aggregate': static_agg,
            'runs': static_runs,
        },
        'contextual': {
            'aggregate': ctx_agg,
            'runs': ctx_runs,
        },
    }

    json_path = os.path.join(OUTPUT_DIR, 'multirun_aggregate.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Aggregate saved to {json_path}")

    # ── Generate report ──
    report = format_report(static_agg, ctx_agg, K)
    report_path = os.path.join(OUTPUT_DIR, 'multirun_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report saved to {report_path}")

    # ── Print report ──
    total_s = time.time() - t_total
    print(f"\n  Total runtime: {total_s / 3600:.1f}h ({total_s:.0f}s)")
    print(f"  Results in: {os.path.abspath(OUTPUT_DIR)}")
    print()
    print(report)


if __name__ == '__main__':
    main()
