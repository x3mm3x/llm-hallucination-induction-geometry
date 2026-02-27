#!/usr/bin/env python3
"""
hallucination_stats.py — Shared statistical testing module
============================================================

Drop-in replacement for the token-level-only `run_statistical_tests()`
found in all four hallucination induction scripts. Fixes the
pseudoreplication problem at source by computing:

  1. Token-level tests  (N ≈ 900) — reported as "inflated" reference
  2. Prompt-level tests  (N = 15)  — the honest primary inference
     - Mann-Whitney U (asymptotic)
     - Permutation p-values (exact or Monte Carlo)
     - BCa bootstrap 95% CIs on rank-biserial effect size
     - Mean AND median aggregation for sensitivity
  3. Holm-Bonferroni correction per test family
  4. Token count diagnostics

Usage in each script:

    from hallucination_stats import (
        extract_prompt_metrics,         # from list-of-dicts format
        extract_prompt_metrics_raw,     # from per-entry hidden_states
        run_two_level_tests,            # the main replacement
        run_band_tests,                 # for spectral per-band analysis
        format_stats_report,            # text report section
        format_band_report,             # text report for spectral bands
        token_diagnostics,              # EOS/token count summary
    )

Designed for N=15 prompts per condition with ~60 tokens each.
"""

import numpy as np
from scipy import stats
from math import comb
from itertools import combinations


# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

SEED = 42  # Locked — all scripts, all stages, one seed

DEFAULT_CONFIG = {
    'n_permutations': 50000,
    'n_bootstrap': 10000,
    'ci_level': 0.95,
    'alpha': 0.05,
    'exact_threshold': 100000,  # use exact perm if C(n,k) <= this
}


# ──────────────────────────────────────────────────────────────────────
# DATA EXTRACTION: Convert script-specific formats to common structure
# ──────────────────────────────────────────────────────────────────────

def extract_prompt_metrics(all_results, metric_names=('H', 'norm', 'max_sim')):
    """Extract per-prompt metric arrays from list-of-dicts format.

    Works with hallucination_induction.py, _contextual.py, _whitened.py
    where each entry has seq['type'], seq['measurements'] (list of dicts).

    Args:
        all_results: list of dicts with 'type' and 'measurements' keys
        metric_names: tuple of metric keys present in each measurement dict

    Returns:
        prompt_data: dict of {condition: {metric: list of per-prompt arrays}}
        token_data:  dict of {condition: {metric: flat np.array}}
        diagnostics: dict of {condition: list of {n_tokens, prompt, hit_eos}}
    """
    conditions = sorted(set(seq['type'] for seq in all_results))

    prompt_data = {c: {m: [] for m in metric_names} for c in conditions}
    token_data = {c: {m: [] for m in metric_names} for c in conditions}
    diagnostics = {c: [] for c in conditions}

    for seq in all_results:
        cond = seq['type']
        measurements = seq.get('measurements', [])
        n_tokens = len(measurements)
        hit_eos = n_tokens < seq.get('_max_tokens', 60)

        diagnostics[cond].append({
            'prompt': seq.get('prompt', ''),
            'n_tokens': n_tokens,
            'hit_eos': hit_eos,
        })

        if n_tokens == 0:
            continue

        for m_name in metric_names:
            vals = np.array([tok[m_name] for tok in measurements])
            prompt_data[cond][m_name].append(vals)
            token_data[cond][m_name].extend(vals.tolist())

    # Convert token data to arrays
    for c in token_data:
        for m in token_data[c]:
            token_data[c][m] = np.array(token_data[c][m])

    return prompt_data, token_data, diagnostics


def extract_prompt_metrics_raw(all_experimental, compute_fn):
    """Extract per-prompt metrics from raw hidden states (spectral script).

    Args:
        all_experimental: list of dicts with 'type', 'hidden_states'
        compute_fn: callable(hidden_states_array) -> dict of {metric: array}
            e.g. computes H, norm, max_sim from whitened hidden states

    Returns:
        prompt_data, token_data, diagnostics (same format as above)
    """
    metric_names = None
    conditions = sorted(set(e['type'] for e in all_experimental))

    prompt_data_tmp = {c: [] for c in conditions}
    diagnostics = {c: [] for c in conditions}

    for entry in all_experimental:
        cond = entry['type']
        hidden = entry['hidden_states']
        n_tokens = len(hidden)

        diagnostics[cond].append({
            'prompt': entry.get('prompt', ''),
            'n_tokens': n_tokens,
            'hit_eos': n_tokens < entry.get('_max_tokens', 60),
        })

        if n_tokens == 0:
            continue

        metrics = compute_fn(hidden)
        if metric_names is None:
            metric_names = list(metrics.keys())

        prompt_data_tmp[cond].append(metrics)

    # Restructure
    prompt_data = {c: {m: [] for m in metric_names} for c in conditions}
    token_data = {c: {m: [] for m in metric_names} for c in conditions}

    for c in conditions:
        for entry_metrics in prompt_data_tmp[c]:
            for m in metric_names:
                vals = np.asarray(entry_metrics[m])
                prompt_data[c][m].append(vals)
                token_data[c][m].extend(vals.tolist())

    for c in token_data:
        for m in token_data[c]:
            token_data[c][m] = np.array(token_data[c][m])

    return prompt_data, token_data, diagnostics


# ──────────────────────────────────────────────────────────────────────
# CORE STATISTICAL FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def _permutation_test_2sample(x, y, n_perm, rng, exact_threshold):
    """Two-sample permutation test on difference in means."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n_x = len(x)
    combined = np.concatenate([x, y])
    n_total = len(combined)
    observed = x.mean() - y.mean()

    if comb(n_total, n_x) <= exact_threshold:
        count = sum(1 for idx in combinations(range(n_total), n_x)
                    if abs(combined[list(idx)].mean() -
                           (combined.sum() - combined[list(idx)].sum()) /
                           (n_total - n_x))
                    >= abs(observed))
        return observed, count / comb(n_total, n_x)
    else:
        count = 0
        for _ in range(n_perm):
            rng.shuffle(combined)
            if abs(combined[:n_x].mean() - combined[n_x:].mean()) >= abs(observed):
                count += 1
        return observed, (count + 1) / (n_perm + 1)


def _permutation_test_kruskal(groups, n_perm, rng):
    """Permutation test using KW H statistic."""
    groups = [np.asarray(g, float) for g in groups]
    sizes = [len(g) for g in groups]
    combined = np.concatenate(groups)
    observed_H, _ = stats.kruskal(*groups)

    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_groups = []
        start = 0
        for s in sizes:
            perm_groups.append(combined[start:start + s])
            start += s
        H, _ = stats.kruskal(*perm_groups)
        if H >= observed_H:
            count += 1
    return observed_H, (count + 1) / (n_perm + 1)


def _rank_biserial(x, y):
    """Rank-biserial r from Mann-Whitney U."""
    U, _ = stats.mannwhitneyu(np.asarray(x), np.asarray(y),
                               alternative='two-sided')
    return 1 - (2 * U) / (len(x) * len(y))


def _bootstrap_effect_ci(x, y, n_boot, ci_level, rng):
    """Bootstrap percentile CI on rank-biserial r."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = _rank_biserial(x, y)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        bx = rng.choice(x, len(x), replace=True)
        by = rng.choice(y, len(y), replace=True)
        try:
            boot[i] = _rank_biserial(bx, by)
        except ValueError:
            boot[i] = 0.0
    alpha = (1 - ci_level) / 2
    return obs, float(np.percentile(boot, 100 * alpha)), \
        float(np.percentile(boot, 100 * (1 - alpha)))


def _bootstrap_diff_ci(x, y, n_boot, ci_level, rng):
    """BCa bootstrap CI on difference in means."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = x.mean() - y.mean()
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = (rng.choice(x, len(x), replace=True).mean() -
                   rng.choice(y, len(y), replace=True).mean())

    # Bias correction
    z0 = stats.norm.ppf(np.clip(np.mean(boot < obs), 1e-10, 1 - 1e-10))

    # Acceleration (jackknife)
    combined = np.concatenate([x, y])
    n = len(combined)
    jack = np.empty(n)
    for i in range(n):
        jx = np.delete(x, i) if i < len(x) else x
        jy = np.delete(y, i - len(x)) if i >= len(x) else y
        jack[i] = jx.mean() - jy.mean()
    jm = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    alpha_lo = (1 - ci_level) / 2
    alpha_hi = 1 - alpha_lo
    for z_a, label in [(stats.norm.ppf(alpha_lo), 'lo'),
                       (stats.norm.ppf(alpha_hi), 'hi')]:
        adj = stats.norm.cdf(z0 + (z0 + z_a) / max(1 - a * (z0 + z_a), 1e-10))
        if label == 'lo':
            ci_lo = np.percentile(boot, 100 * np.clip(adj, 0.001, 0.999))
        else:
            ci_hi = np.percentile(boot, 100 * np.clip(adj, 0.001, 0.999))

    return obs, float(ci_lo), float(ci_hi)


def holm_bonferroni(p_dict, alpha=0.05):
    """Holm-Bonferroni correction.

    Args:
        p_dict: dict of {label: p_value}

    Returns:
        list of (label, raw_p, adj_p, significant) sorted by raw_p
    """
    items = sorted(p_dict.items(), key=lambda x: x[1])
    m = len(items)
    results = []
    for i, (label, p) in enumerate(items):
        adj = min(p * (m - i), 1.0)
        results.append((label, p, adj, adj < alpha))
    # Monotonicity
    for i in range(1, len(results)):
        if results[i][2] < results[i - 1][2]:
            results[i] = (results[i][0], results[i][1],
                          results[i - 1][2], results[i][3])
    return results


# ──────────────────────────────────────────────────────────────────────
# PROMPT-LEVEL AGGREGATION
# ──────────────────────────────────────────────────────────────────────

def _aggregate_to_prompt(prompt_data, agg_fn):
    """Aggregate per-token arrays to single values per prompt.

    Args:
        prompt_data: {condition: {metric: [array_per_prompt, ...]}}
        agg_fn: np.mean or np.median

    Returns:
        {condition: {metric: np.array of prompt-level values}}
    """
    result = {}
    for c in prompt_data:
        result[c] = {}
        for m in prompt_data[c]:
            result[c][m] = np.array([float(agg_fn(arr))
                                     for arr in prompt_data[c][m]
                                     if len(arr) > 0])
    return result


# ──────────────────────────────────────────────────────────────────────
# MAIN TEST RUNNER
# ──────────────────────────────────────────────────────────────────────

def run_two_level_tests(prompt_data, token_data, metric_names=None,
                        conditions=None, cfg=None):
    """Run token-level AND prompt-level tests with full v2 machinery.

    This is the drop-in replacement for each script's run_statistical_tests().

    Args:
        prompt_data: {condition: {metric: [array_per_prompt, ...]}}
        token_data:  {condition: {metric: flat np.array}}
        metric_names: list of metrics to test (default: all in token_data)
        conditions: list of conditions to include (default: all)
        cfg: override DEFAULT_CONFIG

    Returns:
        results: dict with full test results at both levels
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    rng = np.random.default_rng(SEED)

    if conditions is None:
        conditions = sorted(token_data.keys())
    if metric_names is None:
        metric_names = sorted(next(iter(token_data.values())).keys())

    # Prompt-level aggregations
    prompt_mean = _aggregate_to_prompt(prompt_data, np.mean)
    prompt_median = _aggregate_to_prompt(prompt_data, np.median)

    results = {}

    for m_name in metric_names:
        t_groups = [token_data[c][m_name] for c in conditions]
        pm_groups = [prompt_mean[c][m_name] for c in conditions]
        pmed_groups = [prompt_median[c][m_name] for c in conditions]

        m_result = {
            'conditions': conditions,
            'token_n': [len(g) for g in t_groups],
            'prompt_n': [len(g) for g in pm_groups],
        }

        # ── Kruskal-Wallis (3+ groups) ──
        if len(conditions) >= 3:
            if all(len(g) > 2 for g in t_groups):
                kw_H, kw_p = stats.kruskal(*t_groups)
                m_result['token_kw'] = {'H': float(kw_H), 'p': float(kw_p)}

            if all(len(g) > 2 for g in pm_groups):
                kw_H, kw_p = stats.kruskal(*pm_groups)
                m_result['prompt_mean_kw'] = {'H': float(kw_H), 'p': float(kw_p)}

                # Permutation KW
                perm_H, perm_p = _permutation_test_kruskal(
                    pm_groups, cfg['n_permutations'], rng)
                m_result['prompt_mean_kw_perm'] = {
                    'H': float(perm_H), 'p': float(perm_p)}

            if all(len(g) > 2 for g in pmed_groups):
                kw_H, kw_p = stats.kruskal(*pmed_groups)
                m_result['prompt_median_kw'] = {'H': float(kw_H), 'p': float(kw_p)}

        # ── Pairwise ──
        pairwise = {}
        for i, c1 in enumerate(conditions):
            for c2 in conditions[i + 1:]:
                pair_key = f'{c1}_vs_{c2}'
                pw = {}

                # Token level
                g1t, g2t = token_data[c1][m_name], token_data[c2][m_name]
                if len(g1t) > 0 and len(g2t) > 0:
                    U, p = stats.mannwhitneyu(g1t, g2t, alternative='two-sided')
                    r = 1 - (2 * U) / (len(g1t) * len(g2t))
                    pw['token'] = {'p': float(p), 'r': float(r),
                                   'mean_1': float(g1t.mean()),
                                   'mean_2': float(g2t.mean())}

                # Prompt mean — MW
                g1p, g2p = prompt_mean[c1][m_name], prompt_mean[c2][m_name]
                if len(g1p) > 2 and len(g2p) > 2:
                    U, p = stats.mannwhitneyu(g1p, g2p, alternative='two-sided')
                    r = 1 - (2 * U) / (len(g1p) * len(g2p))
                    pw['prompt_mean_mw'] = {'p': float(p), 'r': float(r)}

                    # Permutation
                    _, perm_p = _permutation_test_2sample(
                        g1p, g2p, cfg['n_permutations'], rng,
                        cfg['exact_threshold'])
                    pw['prompt_mean_perm'] = {'p': float(perm_p)}

                    # Bootstrap CI on r
                    r_obs, r_lo, r_hi = _bootstrap_effect_ci(
                        g1p, g2p, cfg['n_bootstrap'], cfg['ci_level'], rng)
                    pw['r_ci'] = {'r': float(r_obs),
                                  'lo': float(r_lo), 'hi': float(r_hi)}

                    # Bootstrap CI on diff means
                    d_obs, d_lo, d_hi = _bootstrap_diff_ci(
                        g1p, g2p, cfg['n_bootstrap'], cfg['ci_level'], rng)
                    pw['diff_ci'] = {'diff': float(d_obs),
                                     'lo': float(d_lo), 'hi': float(d_hi)}

                # Prompt median — MW
                g1m, g2m = prompt_median[c1][m_name], prompt_median[c2][m_name]
                if len(g1m) > 2 and len(g2m) > 2:
                    U, p = stats.mannwhitneyu(g1m, g2m, alternative='two-sided')
                    r = 1 - (2 * U) / (len(g1m) * len(g2m))
                    pw['prompt_median_mw'] = {'p': float(p), 'r': float(r)}

                pairwise[pair_key] = pw

        m_result['pairwise'] = pairwise

        # ── Holm correction on prompt-mean pairwise p-values ──
        holm_input = {}
        for pk, pv in pairwise.items():
            if 'prompt_mean_mw' in pv:
                holm_input[f'{m_name}:{pk}'] = pv['prompt_mean_mw']['p']
        if holm_input:
            m_result['holm'] = holm_bonferroni(holm_input, cfg['alpha'])

        # ── Condition descriptives ──
        m_result['condition_stats'] = {}
        for c in conditions:
            pm_arr = prompt_mean[c][m_name]
            m_result['condition_stats'][c] = {
                'prompt_mean': float(pm_arr.mean()) if len(pm_arr) > 0 else 0,
                'prompt_std': float(pm_arr.std()) if len(pm_arr) > 0 else 0,
                'prompt_median': float(np.median(pm_arr)) if len(pm_arr) > 0 else 0,
                'token_mean': float(token_data[c][m_name].mean())
                    if len(token_data[c][m_name]) > 0 else 0,
                'n_tokens': len(token_data[c][m_name]),
                'n_prompts': len(pm_arr),
            }

        results[m_name] = m_result

    return results


def run_band_tests(all_experimental, band_compute_fn, metric_names,
                   conditions=None, cfg=None):
    """Run two-level tests for a single spectral band.

    Args:
        all_experimental: list of dicts with 'type', 'hidden_states'
        band_compute_fn: callable(hidden_states) -> {metric: array}
        metric_names: list of metric names
        conditions: list of conditions (default: auto-detect)
        cfg: config override

    Returns:
        results dict from run_two_level_tests
    """
    prompt_data, token_data, diagnostics = extract_prompt_metrics_raw(
        all_experimental, band_compute_fn)

    if conditions is None:
        conditions = sorted(token_data.keys())

    return run_two_level_tests(prompt_data, token_data,
                               metric_names=metric_names,
                               conditions=conditions, cfg=cfg)


# ──────────────────────────────────────────────────────────────────────
# TOKEN DIAGNOSTICS
# ──────────────────────────────────────────────────────────────────────

def token_diagnostics(diagnostics):
    """Generate token count diagnostics text.

    Args:
        diagnostics: {condition: [{n_tokens, prompt, hit_eos}, ...]}

    Returns:
        list of text lines
    """
    lines = ['TOKEN COUNT DIAGNOSTICS', '-' * 60]
    for c in sorted(diagnostics.keys()):
        entries = diagnostics[c]
        counts = np.array([e['n_tokens'] for e in entries])
        n_eos = sum(1 for e in entries if e['hit_eos'])
        lines.append(
            f"  {c}: N={len(entries)}, tokens/prompt: "
            f"mean={counts.mean():.1f}, median={np.median(counts):.0f}, "
            f"min={counts.min()}, max={counts.max()}, "
            f"EOS early={n_eos}/{len(entries)}")
    lines.append('')
    return lines


# ──────────────────────────────────────────────────────────────────────
# REPORT FORMATTING
# ──────────────────────────────────────────────────────────────────────

def format_stats_report(results, title="STATISTICAL TESTS"):
    """Format two-level test results as report text.

    Args:
        results: output of run_two_level_tests()
        title: section title

    Returns:
        list of text lines
    """
    lines = []
    lines.append(f"\n{'─' * 90}")
    lines.append(title)
    lines.append(f"{'─' * 90}")
    lines.append("")
    lines.append("  Two-level analysis: token-level (inflated N) and "
                 "prompt-level (conservative N=15)")
    lines.append("  Primary inference: prompt-level permutation p-values "
                 "with bootstrap CIs on effect sizes")
    lines.append("")

    for m_name, m_res in results.items():
        conditions = m_res['conditions']
        n_conds = len(conditions)

        lines.append(f"  ── {m_name} {'─' * 70}")

        # KW tests
        if 'token_kw' in m_res:
            tkw = m_res['token_kw']
            lines.append(f"    Kruskal-Wallis (token-level):  "
                         f"H={tkw['H']:.2f}, p={tkw['p']:.2e}")
        if 'prompt_mean_kw' in m_res:
            pkw = m_res['prompt_mean_kw']
            ppkw = m_res.get('prompt_mean_kw_perm', {})
            mkw = m_res.get('prompt_median_kw', {})
            lines.append(
                f"    Kruskal-Wallis (prompt-mean):  "
                f"H={pkw['H']:.2f}, p={pkw['p']:.2e}  "
                f"Perm p={ppkw.get('p', float('nan')):.2e}  "
                f"Median p={mkw.get('p', float('nan')):.2e}")

        # Condition descriptives
        lines.append(f"    {'Condition':<12} {'N_tok':>6} {'N_prm':>6} "
                     f"{'Tok mean':>10} {'Prm mean':>10} {'Prm med':>10}")
        lines.append(f"    {'-' * 58}")
        for c in conditions:
            cs = m_res['condition_stats'][c]
            lines.append(
                f"    {c:<12} {cs['n_tokens']:>6} {cs['n_prompts']:>6} "
                f"{cs['token_mean']:>10.4f} {cs['prompt_mean']:>10.4f} "
                f"{cs['prompt_median']:>10.4f}")

        # Pairwise
        lines.append("")
        lines.append(
            f"    {'Pair':<18} {'Tok p':>10} {'MW p':>10} {'Perm p':>10} "
            f"{'Med p':>10} {'r':>7} {'r 95% CI':>20}")
        lines.append(f"    {'-' * 88}")

        for pair_key, pw in m_res['pairwise'].items():
            tok_p = pw.get('token', {}).get('p', float('nan'))
            mw_p = pw.get('prompt_mean_mw', {}).get('p', float('nan'))
            perm_p = pw.get('prompt_mean_perm', {}).get('p', float('nan'))
            med_p = pw.get('prompt_median_mw', {}).get('p', float('nan'))
            r_ci = pw.get('r_ci', {})
            r = r_ci.get('r', float('nan'))
            r_lo = r_ci.get('lo', float('nan'))
            r_hi = r_ci.get('hi', float('nan'))

            sig = '*' if mw_p < 0.05 else ''
            lines.append(
                f"    {pair_key:<18} {tok_p:>10.2e} {mw_p:>10.4f} "
                f"{perm_p:>10.4f} {med_p:>10.4f} {r:>+7.3f} "
                f"[{r_lo:>+7.3f}, {r_hi:>+7.3f}] {sig}")

        # Holm correction
        if 'holm' in m_res:
            lines.append("")
            lines.append(f"    Holm-Bonferroni correction:")
            lines.append(f"    {'Test':<30} {'Raw p':>10} {'Adj p':>10} {'Sig':>5}")
            lines.append(f"    {'-' * 57}")
            for label, raw_p, adj_p, sig in m_res['holm']:
                lines.append(
                    f"    {label:<30} {raw_p:>10.4f} {adj_p:>10.4f} "
                    f"{'YES' if sig else 'no':>5}")

        lines.append("")

    return lines


def format_band_report(band_label, band_results, metric_names,
                       pair_filter=None):
    """Format per-band test results compactly.

    Args:
        band_label: e.g. "PC 129-256 (mid-B)"
        band_results: output of run_two_level_tests for one band
        metric_names: list of metrics to report
        pair_filter: optional list of pair keys to show
            (default: all)

    Returns:
        list of text lines
    """
    lines = [f"\n  {band_label}:"]

    for m_name in metric_names:
        m_res = band_results.get(m_name, {})
        if not isinstance(m_res, dict) or 'pairwise' not in m_res:
            continue
        pairwise = m_res['pairwise']
        pairs = pair_filter or list(pairwise.keys())

        for pk in pairs:
            pw = pairwise.get(pk, {})
            if not pw:
                continue

            tok_p = pw.get('token', {}).get('p', float('nan'))
            mw_p = pw.get('prompt_mean_mw', {}).get('p', float('nan'))
            perm_p = pw.get('prompt_mean_perm', {}).get('p', float('nan'))
            med_p = pw.get('prompt_median_mw', {}).get('p', float('nan'))
            r_ci = pw.get('r_ci', {})
            r = r_ci.get('r', float('nan'))
            r_lo = r_ci.get('lo', float('nan'))
            r_hi = r_ci.get('hi', float('nan'))
            surv = "YES" if mw_p < 0.05 else "no"

            lines.append(
                f"    {m_name:<8} {pk:<20} Tok={tok_p:.2e} "
                f"MW={mw_p:.4f} Perm={perm_p:.4f} Med={med_p:.4f} "
                f"r={r:+.3f} [{r_lo:+.3f},{r_hi:+.3f}] {surv}")

    return lines


def format_survivors_summary(all_band_results, band_labels, alpha=0.05):
    """Summarize all surviving band results across the full matrix.

    Args:
        all_band_results: list of run_two_level_tests outputs (one per band)
        band_labels: list of band label strings
        alpha: significance threshold

    Returns:
        list of text lines, list of surviving (band, metric, pair, pw) tuples
    """
    survivors = []
    for band_res, label in zip(all_band_results, band_labels):
        for m_name, m_res in band_res.items():
            if not isinstance(m_res, dict) or 'pairwise' not in m_res:
                continue
            for pk, pw in m_res['pairwise'].items():
                mw = pw.get('prompt_mean_mw', {})
                if mw.get('p', 1.0) < alpha:
                    survivors.append((label, m_name, pk, pw))

    lines = []
    lines.append(f"\n{'─' * 90}")
    lines.append(f"ALL SURVIVING PROMPT-LEVEL RESULTS (MW p < {alpha})")
    lines.append(f"{'─' * 90}")
    lines.append(f"  {'Band':<28} {'Metric':<8} {'Pair':<20} "
                 f"{'MW p':>8} {'Perm p':>8} {'Med p':>8} "
                 f"{'r':>7} {'r CI':>20}")
    lines.append(f"  {'-' * 105}")

    for label, m_name, pk, pw in survivors:
        mw_p = pw['prompt_mean_mw']['p']
        perm_p = pw.get('prompt_mean_perm', {}).get('p', float('nan'))
        med_p = pw.get('prompt_median_mw', {}).get('p', float('nan'))
        r_ci = pw.get('r_ci', {})
        r = r_ci.get('r', float('nan'))
        r_lo = r_ci.get('lo', float('nan'))
        r_hi = r_ci.get('hi', float('nan'))

        # Concordance flag
        all_agree = (mw_p < alpha and perm_p < alpha and med_p < alpha)
        flag = '✓✓✓' if all_agree else ('✓✓' if perm_p < alpha else '✓')

        lines.append(
            f"  {label:<28} {m_name:<8} {pk:<20} "
            f"{mw_p:>8.4f} {perm_p:>8.4f} {med_p:>8.4f} "
            f"{r:>+7.3f} [{r_lo:>+.3f},{r_hi:>+.3f}] {flag}")

    lines.append("")
    lines.append(f"  Total surviving: {len(survivors)}  "
                 f"(✓✓✓ = MW+Perm+Med agree, "
                 f"✓✓ = MW+Perm agree, ✓ = MW only)")
    lines.append("")

    return lines, survivors


# ──────────────────────────────────────────────────────────────────────
# JSON SERIALIZATION HELPER
# ──────────────────────────────────────────────────────────────────────

def results_to_json(results):
    """Convert run_two_level_tests output to JSON-safe dict."""
    import copy

    def _clean(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        return obj

    return _clean(results)
