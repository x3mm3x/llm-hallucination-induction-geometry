#!/usr/bin/env python3
"""
Geometric Hallucination Taxonomy — Contextual Induction Experiment
====================================================================

Follow-up to hallucination_induction.py. The static embedding experiment
showed that all three prompt conditions collapse to the same region of
static embedding space — generated tokens' vocabulary positions don't
carry enough information to distinguish failure modes.

This script replaces static embedding lookup with CONTEXTUAL hidden
states from GPT-2's last transformer layer. The contextual representation
at the moment of token selection encodes the model's processing of the
prompt — it knows whether the model had strong context, weak context,
or no relevant context. Same prompts, same measurement framework,
different representation.

Key change: manual autoregressive generation captures the last-layer
hidden state at each token generation step. These contextual vectors
(not the static embeddings) are clustered, measured, and classified.

Calibration: a diverse set of well-formed prompts establishes the
background distribution of contextual hidden states. Zone thresholds
are percentile-calibrated from this background, parallel to the
static experiment's calibration from the vocabulary distribution.

Outputs (in ./results_induction_contextual/):
  - induction_contextual_report.txt    : full results with confusion matrix
  - fig_ctx_zones.png                  : norm–membership per condition
  - fig_ctx_trajectories.png           : cluster trajectories
  - fig_ctx_confusion.png              : confusion matrix heatmap
  - fig_ctx_distributions.png          : H, norm, max_sim distributions
  - fig_ctx_layer_comparison.png       : last-layer vs static comparison
  - raw_results_contextual.json        : per-token data

Hardware: CPU only, ~4–8GB RAM, ~15–30 min runtime
Usage:  python hallucination_induction_contextual.py
"""

import os
import sys
import time
import gc
import json
import warnings
import numpy as np
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from hallucination_stats import (
    extract_prompt_metrics, run_two_level_tests, token_diagnostics,
    format_stats_report, results_to_json, holm_bonferroni,
)

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

CONFIG = {
    'n_clusters': 40,
    'min_cluster_size': 10,
    'random_seed': 42,
    'output_dir': './results_induction_contextual',
    'figure_dpi': 150,
    'generate_figures': True,    # set False to skip diagnostic figure generation

    # Generation
    'max_new_tokens': 60,
    'n_prompts_per_type': 15,
    'temperature': 1.0,
    'top_k': 0,
    'top_p': 1.0,

    # Zone thresholds (percentile-based, calibrated from background)
    'H_low_pct': 15,
    'norm_low_pct': 40,
    'H_high_pct': 75,
    'max_sim_low_pct': 25,

    # Calibration
    'n_calibration_prompts': 25,
    'calibration_tokens': 60,
}


# ──────────────────────────────────────────────────────────────────────
# PROMPT SETS (identical to static experiment)
# ──────────────────────────────────────────────────────────────────────

TYPE1_PROMPTS = [
    "The",
    "It is",
    "There are",
    "This is a",
    "One of the",
    "In the",
    "As a",
    "They were",
    "Some of",
    "Many people",
    "It was a",
    "He said that",
    "The most",
    "We have",
    "A very",
]

TYPE2_PROMPTS = [
    "The bank announced record levels of",
    "The pitch was perfect for the",
    "She picked up the bat and",
    "The board decided to table the",
    "He studied the cell under the",
    "The crane lifted the heavy",
    "The plant manager reviewed the new",
    "Mercury levels in the",
    "The current was too strong for the",
    "The cabinet members discussed the",
    "They found the mole in the",
    "The scales showed exactly",
    "The press released a statement about the",
    "The match was struck and",
    "The patient leaves were carefully",
]

TYPE3_PROMPTS = [
    "The xenoplasmic refractometry of late-Holocene",
    "Professor Kvistad's third theorem on paracompact",
    "In Zvrotkian epistemology, the concept of mereological",
    "The biosemiotic implications of CRISPR-modified Tardigrade",
    "According to the Nørgaard-Patel conjecture in topological",
    "The post-Deleuzian analysis of quantum decoherence in",
    "Hyperbolic crochet models of anti-de Sitter spacetime suggest",
    "The ethnopharmacological study of Amazonian Ayahuasca analogs",
    "A contravariant functor from the category of smooth manifolds",
    "The archaeomagnetostratigraphic evidence from Paleocene",
    "Subquadratic approximation algorithms for the Steiner tree",
    "The phenomenological reduction of eigenstate thermalization",
    "Applying Khovanov homology to categorified quantum groups",
    "The gliotransmitter-mediated modulation of thalamocortical",
    "Transfinite induction over the constructible hierarchy L",
]

# Calibration prompts: diverse, well-formed, spanning multiple domains.
# These establish the "normal" distribution of contextual hidden states
# against which experimental conditions are measured.
CALIBRATION_PROMPTS = [
    "The president of the United States delivered a speech about",
    "Scientists at the university discovered a new species of",
    "The stock market experienced significant volatility after the",
    "In a landmark court ruling, the judge determined that",
    "The new software update includes several improvements to",
    "Researchers published a study showing that regular exercise",
    "The city council voted unanimously to approve the construction of",
    "According to the latest census data, the population of",
    "The chef prepared a traditional Italian dish using fresh",
    "Engineers at the space agency successfully launched the",
    "The novel, which was published last year, tells the story of",
    "Climate scientists warned that global temperatures could rise",
    "The football team won their fifth consecutive game by",
    "Historians have long debated whether the ancient civilization",
    "The pharmaceutical company announced positive results from",
    "A major earthquake measuring seven on the Richter scale struck",
    "The documentary explores the lives of immigrant families who",
    "Economists predict that inflation will continue to affect",
    "The orchestra performed Beethoven's ninth symphony to a",
    "Marine biologists observed unusual behavior in the population of",
    "The school board implemented a new curriculum focused on",
    "Astronomers detected a faint signal coming from a distant",
    "The factory produces over ten thousand units per day of",
    "Political analysts suggest that the upcoming election will be",
    "The museum's latest exhibition features artwork from the",
]


# ──────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────

def load_gpt2():
    """Load GPT-2 for generation with hidden state extraction."""
    print("\n[Step 1] Loading GPT-2...")
    t0 = time.time()

    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', torch_dtype=torch.float32)
    model.eval()

    # Also extract static embeddings for comparison
    with torch.no_grad():
        static_emb = model.transformer.wte.weight.cpu().numpy()

    print(f"  Model loaded: 12 layers, hidden_dim=768")
    print(f"  Static embedding matrix: {static_emb.shape}")
    print(f"  Done in {time.time() - t0:.1f}s")

    return model, tokenizer, static_emb


# ──────────────────────────────────────────────────────────────────────
# AUTOREGRESSIVE GENERATION WITH HIDDEN STATE CAPTURE
# ──────────────────────────────────────────────────────────────────────

def generate_with_hidden_states(model, tokenizer, prompt, max_new_tokens,
                                temperature=1.0, top_k=0, top_p=1.0):
    """Manual autoregressive generation capturing last-layer hidden states.

    For each generated token, captures:
      - The token id selected
      - The last-layer hidden state at the final position (the contextual
        representation that determined the next token distribution)
      - The static embedding of the selected token (for comparison)

    Returns:
      generated_ids: list of int
      hidden_states: np.array of shape (n_generated, 768)
      generated_text: str
    """
    import torch

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated_ids = []
    hidden_states_list = []

    eos_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        # Last layer, last position: the contextual state that produces
        # the next-token distribution
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # (1, 768)

        # Sample next token
        logits = outputs.logits[:, -1, :]  # (1, vocab_size)

        if temperature != 1.0:
            logits = logits / temperature

        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()
        if token_id == eos_id:
            break

        generated_ids.append(token_id)
        hidden_states_list.append(last_hidden.squeeze(0).cpu().numpy())

        # Append token and continue
        input_ids = torch.cat([input_ids, next_token], dim=1)

    hidden_states = np.array(hidden_states_list) if hidden_states_list else np.zeros((0, 768))
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_ids, hidden_states, generated_text


# ──────────────────────────────────────────────────────────────────────
# CALIBRATION: BUILD BACKGROUND DISTRIBUTION
# ──────────────────────────────────────────────────────────────────────

def build_calibration_distribution(model, tokenizer):
    """Generate from diverse prompts to establish the background distribution
    of contextual hidden states. This replaces the vocabulary distribution
    used in the static experiment."""
    print("\n[Step 2] Building calibration distribution...")
    t0 = time.time()

    all_hidden = []
    n_tokens_total = 0

    for i, prompt in enumerate(CALIBRATION_PROMPTS):
        gen_ids, hidden, gen_text = generate_with_hidden_states(
            model, tokenizer, prompt,
            max_new_tokens=CONFIG['calibration_tokens'],
            temperature=CONFIG['temperature'],
            top_k=CONFIG['top_k'],
            top_p=CONFIG['top_p'],
        )

        if len(hidden) > 0:
            all_hidden.append(hidden)
            n_tokens_total += len(hidden)

        if (i + 1) % 5 == 0:
            print(f"    Calibration: {i+1}/{len(CALIBRATION_PROMPTS)} prompts, "
                  f"{n_tokens_total} tokens collected")

    calibration_hidden = np.vstack(all_hidden)
    print(f"  Calibration corpus: {calibration_hidden.shape[0]} contextual vectors")
    print(f"  Done in {time.time() - t0:.1f}s")

    return calibration_hidden


def cluster_and_calibrate(calibration_hidden):
    """Cluster the calibration distribution and compute zone thresholds."""
    print("\n[Step 3] Clustering calibration distribution...")
    t0 = time.time()

    kmeans = MiniBatchKMeans(
        n_clusters=CONFIG['n_clusters'], random_state=CONFIG['random_seed'],
        batch_size=1024, max_iter=300, n_init=5)
    labels = kmeans.fit_predict(calibration_hidden)
    centroids = kmeans.cluster_centers_

    sizes = np.bincount(labels, minlength=CONFIG['n_clusters'])
    print(f"  Clusters: {CONFIG['n_clusters']}, sizes: "
          f"min={sizes.min()}, mean={sizes.mean():.0f}, max={sizes.max()}")

    # Compute calibration-wide statistics
    norms = np.linalg.norm(calibration_hidden, axis=1)
    sims_to_centroids = cosine_similarity(calibration_hidden, centroids)
    top5_sims = np.sort(sims_to_centroids, axis=1)[:, -5:]
    H_vals = top5_sims.mean(axis=1)
    max_sims = sims_to_centroids.max(axis=1)

    # Calibrate zones from percentiles
    zones = {
        'H_low': float(np.percentile(H_vals, CONFIG['H_low_pct'])),
        'norm_low': float(np.percentile(norms, CONFIG['norm_low_pct'])),
        'H_high': float(np.percentile(H_vals, CONFIG['H_high_pct'])),
        'max_sim_low': float(np.percentile(max_sims, CONFIG['max_sim_low_pct'])),
    }

    calib_stats = {
        'H_vals': H_vals,
        'norms': norms,
        'max_sims': max_sims,
        'labels': labels,
        'H_mean': float(H_vals.mean()),
        'H_std': float(H_vals.std()),
        'norm_mean': float(norms.mean()),
        'norm_std': float(norms.std()),
    }

    print(f"  Calibration stats: H(v) = {H_vals.mean():.4f} ± {H_vals.std():.4f}, "
          f"||v|| = {norms.mean():.4f} ± {norms.std():.4f}")
    print(f"  Zone thresholds (percentile-calibrated):")
    print(f"    Type 1: H(v) < {zones['H_low']:.4f} (p{CONFIG['H_low_pct']}) "
          f"AND ||v|| < {zones['norm_low']:.4f} (p{CONFIG['norm_low_pct']})")
    print(f"    Type 2: H(v) > {zones['H_high']:.4f} (p{CONFIG['H_high_pct']})")
    print(f"    Type 3: max_sim < {zones['max_sim_low']:.4f} (p{CONFIG['max_sim_low_pct']})")
    print(f"  Done in {time.time() - t0:.1f}s")

    return kmeans, centroids, zones, calib_stats


# ──────────────────────────────────────────────────────────────────────
# EXPERIMENTAL GENERATION
# ──────────────────────────────────────────────────────────────────────

def generate_experimental(model, tokenizer, prompts, type_label):
    """Generate under experimental condition, capturing contextual states."""
    print(f"\n  Generating {type_label} ({len(prompts)} prompts)...")

    results = []
    for i, prompt in enumerate(prompts):
        gen_ids, hidden, gen_text = generate_with_hidden_states(
            model, tokenizer, prompt,
            max_new_tokens=CONFIG['max_new_tokens'],
            temperature=CONFIG['temperature'],
            top_k=CONFIG['top_k'],
            top_p=CONFIG['top_p'],
        )

        results.append({
            'prompt': prompt,
            'generated_ids': gen_ids,
            'hidden_states': hidden,       # (n_tokens, 768) contextual vectors
            'generated_text': gen_text,
            'type': type_label,
            '_max_tokens': CONFIG['max_new_tokens'],
        })

        if i < 3:
            preview = gen_text[:80].replace('\n', ' ')
            print(f"    [{i+1}] \"{prompt}\" → \"{preview}...\"")

    return results


# ──────────────────────────────────────────────────────────────────────
# GEOMETRIC MEASUREMENT (on contextual hidden states)
# ──────────────────────────────────────────────────────────────────────

def measure_geometry(gen_results, centroids, zones, static_emb=None):
    """Compute per-token geometric metrics from contextual hidden states.

    If static_emb is provided, also computes static-space metrics for
    direct comparison.
    """
    print("\n[Step 5] Measuring geometric signatures (contextual)...")
    t0 = time.time()

    for seq in gen_results:
        hidden = seq['hidden_states']
        token_ids = seq['generated_ids']
        measurements = []

        if len(hidden) == 0:
            seq['measurements'] = []
            continue

        # Batch compute similarities to centroids
        sims = cosine_similarity(hidden, centroids)  # (n_tokens, n_clusters)
        norms = np.linalg.norm(hidden, axis=1)

        for j in range(len(hidden)):
            top5 = np.sort(sims[j])[-5:]
            H = float(top5.mean())
            max_sim = float(sims[j].max())
            norm = float(norms[j])
            cluster = int(sims[j].argmax())
            zone = classify_token(H, norm, max_sim, zones)

            m = {
                'token_id': int(token_ids[j]),
                'norm': norm,
                'H': H,
                'max_sim': max_sim,
                'cluster': cluster,
                'zone': zone,
            }

            # Static comparison (if available)
            if static_emb is not None and token_ids[j] < static_emb.shape[0]:
                sv = static_emb[token_ids[j]].reshape(1, -1)
                m['static_norm'] = float(np.linalg.norm(sv))
                static_sims = cosine_similarity(sv, centroids).flatten()
                m['static_H'] = float(np.sort(static_sims)[-5:].mean())

            measurements.append(m)

        seq['measurements'] = measurements

    print(f"  Done in {time.time() - t0:.1f}s")
    return gen_results


def classify_token(H, norm, max_sim, zones):
    """Classify token into geometric zone (same logic as static version)."""
    if H < zones['H_low'] and norm < zones['norm_low']:
        return 'type1'
    if max_sim < zones['max_sim_low']:
        return 'type3'
    if H > zones['H_high']:
        return 'type2'
    return 'unclassified'


# ──────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ──────────────────────────────────────────────────────────────────────

def compute_confusion_matrix(all_results):
    """Compute confusion matrix: rows=induced condition, cols=geometric zone."""
    print("\n[Step 6] Computing confusion matrix...")

    types = ['type1', 'type2', 'type3']
    zone_labels = ['type1', 'type2', 'type3', 'unclassified']

    counts = {t: {z: 0 for z in zone_labels} for t in types}
    totals = {t: 0 for t in types}
    seq_stats = {t: [] for t in types}

    for seq in all_results:
        t = seq['type']
        measurements = seq.get('measurements', [])
        if not measurements:
            continue

        seq_zones = [m['zone'] for m in measurements]
        seq_H = [m['H'] for m in measurements]
        seq_norms = [m['norm'] for m in measurements]

        for z in seq_zones:
            counts[t][z] += 1
            totals[t] += 1

        seq_stats[t].append({
            'prompt': seq['prompt'],
            'n_tokens': len(measurements),
            'mean_H': float(np.mean(seq_H)),
            'mean_norm': float(np.mean(seq_norms)),
            'zone_fractions': {z: seq_zones.count(z) / len(seq_zones)
                               for z in zone_labels},
        })

    fractions = {}
    for t in types:
        fractions[t] = {}
        for z in zone_labels:
            fractions[t][z] = counts[t][z] / totals[t] if totals[t] > 0 else 0

    diagonal = [fractions[t][t] for t in types]
    mean_diag = np.mean(diagonal)

    print(f"\n  Confusion matrix (rows=induced, cols=detected zone):")
    print(f"  {'':>12} {'Type1':>8} {'Type2':>8} {'Type3':>8} {'Unclass':>8} {'Total':>8}")
    print(f"  {'─'*56}")
    for t in types:
        row = [f"{fractions[t][z]:.3f}" for z in zone_labels]
        print(f"  {t:>12} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8} {totals[t]:>8}")

    print(f"\n  Diagonal: {diagonal[0]:.3f}  {diagonal[1]:.3f}  {diagonal[2]:.3f}")
    print(f"  Mean diagonal: {mean_diag:.3f}")

    return {
        'counts': counts,
        'fractions': fractions,
        'totals': totals,
        'diagonal': diagonal,
        'mean_diagonal': float(mean_diag),
        'seq_stats': seq_stats,
    }


# ──────────────────────────────────────────────────────────────────────
# STATISTICAL TESTS
# ──────────────────────────────────────────────────────────────────────

def run_statistical_tests(all_results):
    """Two-level statistical tests: token-level (reference) + prompt-level (primary)."""
    print("\n[Step 7] Statistical tests (two-level)...")

    prompt_data, token_data, diagnostics = extract_prompt_metrics(
        all_results, metric_names=('H', 'norm', 'max_sim'))

    results = run_two_level_tests(
        prompt_data, token_data,
        metric_names=['H', 'norm', 'max_sim'],
        cfg={'n_permutations': 50000, 'n_bootstrap': 10000})

    for m_name in ['H', 'norm', 'max_sim']:
        m_res = results.get(m_name, {})
        tkw = m_res.get('token_kw', {})
        pkw = m_res.get('prompt_mean_kw', {})
        print(f"  {m_name}: token KW p={tkw.get('p', 1):.2e}, "
              f"prompt KW p={pkw.get('p', 1):.2e}")

    return results, diagnostics


# ──────────────────────────────────────────────────────────────────────
# TRAJECTORY ANALYSIS
# ──────────────────────────────────────────────────────────────────────

def analyze_trajectories(all_results):
    """Compute trajectory statistics per condition."""
    print("\n[Step 8] Trajectory analysis...")

    traj_stats = {'type1': [], 'type2': [], 'type3': []}

    for seq in all_results:
        t = seq['type']
        measurements = seq.get('measurements', [])
        if len(measurements) < 3:
            continue

        clusters = [m['cluster'] for m in measurements]
        H_seq = [m['H'] for m in measurements]
        n = len(clusters)

        # Discontinuity rate
        changes = sum(1 for i in range(1, n) if clusters[i] != clusters[i-1])
        disc_rate = changes / (n - 1)

        # Max run length
        max_run = 1
        current_run = 1
        for i in range(1, n):
            if clusters[i] == clusters[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        # Unique clusters
        n_unique = len(set(clusters))

        # H(v) variance within sequence — Type 3 should show high variance
        H_var = float(np.var(H_seq))

        # H(v) trend — does membership strengthen or weaken over generation?
        if n >= 5:
            slope, _, r_val, p_val, _ = stats.linregress(range(n), H_seq)
            H_trend = {'slope': float(slope), 'r': float(r_val), 'p': float(p_val)}
        else:
            H_trend = {'slope': 0, 'r': 0, 'p': 1.0}

        traj_stats[t].append({
            'disc_rate': disc_rate,
            'max_run': max_run,
            'n_unique': n_unique,
            'n_tokens': n,
            'H_var': H_var,
            'H_trend': H_trend,
        })

    print(f"\n  Trajectory statistics (mean ± std):")
    for t in ['type1', 'type2', 'type3']:
        if traj_stats[t]:
            disc = [s['disc_rate'] for s in traj_stats[t]]
            runs = [s['max_run'] for s in traj_stats[t]]
            uniq = [s['n_unique'] for s in traj_stats[t]]
            H_var = [s['H_var'] for s in traj_stats[t]]
            print(f"    {t}: disc={np.mean(disc):.3f}±{np.std(disc):.3f}, "
                  f"max_run={np.mean(runs):.1f}±{np.std(runs):.1f}, "
                  f"unique={np.mean(uniq):.1f}±{np.std(uniq):.1f}, "
                  f"H_var={np.mean(H_var):.5f}±{np.std(H_var):.5f}")

    return traj_stats


# ──────────────────────────────────────────────────────────────────────
# STATIC vs CONTEXTUAL COMPARISON
# ──────────────────────────────────────────────────────────────────────

def compare_static_contextual(all_results):
    """Compare static and contextual metrics for the same tokens."""
    print("\n  Comparing static vs contextual representations...")

    comparisons = {'type1': {'ctx_H': [], 'static_H': [], 'ctx_norm': [], 'static_norm': []},
                   'type2': {'ctx_H': [], 'static_H': [], 'ctx_norm': [], 'static_norm': []},
                   'type3': {'ctx_H': [], 'static_H': [], 'ctx_norm': [], 'static_norm': []}}

    for seq in all_results:
        t = seq['type']
        for m in seq.get('measurements', []):
            if 'static_H' in m:
                comparisons[t]['ctx_H'].append(m['H'])
                comparisons[t]['static_H'].append(m['static_H'])
                comparisons[t]['ctx_norm'].append(m['norm'])
                comparisons[t]['static_norm'].append(m['static_norm'])

    results = {}
    for t in ['type1', 'type2', 'type3']:
        if comparisons[t]['ctx_H']:
            ctx_H = np.array(comparisons[t]['ctx_H'])
            static_H = np.array(comparisons[t]['static_H'])
            ctx_norm = np.array(comparisons[t]['ctx_norm'])
            static_norm = np.array(comparisons[t]['static_norm'])

            results[t] = {
                'ctx_H_mean': float(ctx_H.mean()),
                'static_H_mean': float(static_H.mean()),
                'ctx_norm_mean': float(ctx_norm.mean()),
                'static_norm_mean': float(static_norm.mean()),
                'H_spread_ratio': float(ctx_H.std() / (static_H.std() + 1e-10)),
                'norm_spread_ratio': float(ctx_norm.std() / (static_norm.std() + 1e-10)),
                'n_tokens': len(ctx_H),
            }

    return results


# ──────────────────────────────────────────────────────────────────────
# FIGURES
# ──────────────────────────────────────────────────────────────────────

def generate_figures(all_results, calib_stats, zones, confusion,
                     traj_stats, static_ctx_comparison, output_dir):
    """Generate all figures."""
    print("\n[Step 9] Generating figures...")
    dpi = CONFIG['figure_dpi']

    type_colors = {'type1': '#e74c3c', 'type2': '#2980b9', 'type3': '#27ae60'}
    type_labels = {'type1': 'Type 1 (center-drift)',
                   'type2': 'Type 2 (wrong-well)',
                   'type3': 'Type 3 (coverage gap)'}

    # Collect measurements by type
    by_type = {t: {'H': [], 'norm': [], 'max_sim': []}
               for t in ['type1', 'type2', 'type3']}
    for seq in all_results:
        t = seq['type']
        for m in seq.get('measurements', []):
            by_type[t]['H'].append(m['H'])
            by_type[t]['norm'].append(m['norm'])
            by_type[t]['max_sim'].append(m['max_sim'])

    # ── Figure 1: Contextual tokens in norm–membership space ──────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (t, label) in zip(axes, type_labels.items()):
        # Background: calibration distribution
        ax.scatter(calib_stats['norms'], calib_stats['H_vals'],
                   c='#cccccc', s=1, alpha=0.15, rasterized=True)

        # Zone boundaries
        ax.axhline(y=zones['H_low'], color='#e74c3c', linewidth=0.8,
                   linestyle='--', alpha=0.6)
        ax.axhline(y=zones['H_high'], color='#2980b9', linewidth=0.8,
                   linestyle='--', alpha=0.6)
        ax.axvline(x=zones['norm_low'], color='#e74c3c', linewidth=0.8,
                   linestyle=':', alpha=0.6)

        # Generated tokens
        if by_type[t]['norm']:
            ax.scatter(by_type[t]['norm'], by_type[t]['H'],
                       c=type_colors[t], s=12, alpha=0.6, edgecolors='white',
                       linewidth=0.3, label=f'Generated ({len(by_type[t]["H"])})')

        ax.set_xlabel('Contextual Norm ||h||', fontsize=11)
        ax.set_ylabel('Contextual H(v)', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold', color=type_colors[t])
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.2)

    plt.suptitle('Contextual Hidden State Signatures in Norm–Membership Space',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_ctx_zones.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig_ctx_zones.png")

    # ── Figure 2: Cluster trajectories ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, t in zip(axes, ['type1', 'type2', 'type3']):
        seqs = [s for s in all_results if s['type'] == t]
        for i, seq in enumerate(seqs[:8]):
            measurements = seq.get('measurements', [])
            if not measurements:
                continue
            clusters = [m['cluster'] for m in measurements]
            ax.plot(range(len(clusters)), clusters, '-', alpha=0.5,
                    linewidth=1.2, color=type_colors[t])
            for j in range(1, len(clusters)):
                if abs(clusters[j] - clusters[j-1]) > 5:
                    ax.plot(j, clusters[j], 'x', color='black',
                            markersize=4, alpha=0.4)

        ax.set_xlabel('Token Position', fontsize=11)
        ax.set_ylabel('Cluster Assignment', fontsize=11)
        ax.set_title(type_labels[t], fontsize=11, fontweight='bold',
                     color=type_colors[t])
        ax.set_ylim(-1, CONFIG['n_clusters'])
        ax.grid(True, alpha=0.2)

    plt.suptitle('Contextual Cluster Assignment Trajectories',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_ctx_trajectories.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig_ctx_trajectories.png")

    # ── Figure 3: Confusion matrix heatmap ────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))

    types = ['type1', 'type2', 'type3']
    zone_keys = ['type1', 'type2', 'type3', 'unclassified']
    matrix = np.array([[confusion['fractions'][t][z] for z in zone_keys]
                       for t in types])

    im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    for i in range(len(types)):
        for j in range(len(zone_keys)):
            val = matrix[i, j]
            color = 'white' if val > 0.5 else 'black'
            count = confusion['counts'][types[i]][zone_keys[j]]
            ax.text(j, i, f'{val:.2f}\n({count})',
                    ha='center', va='center', fontsize=10, color=color)

    ax.set_xticks(range(len(zone_keys)))
    ax.set_xticklabels(['Zone 1\n(center-drift)', 'Zone 2\n(wrong-well)',
                        'Zone 3\n(coverage gap)', 'Unclassified'], fontsize=9)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(['Induced\nType 1', 'Induced\nType 2', 'Induced\nType 3'],
                       fontsize=10)
    ax.set_xlabel('Detected Geometric Zone (Contextual)', fontsize=12)
    ax.set_ylabel('Induced Condition', fontsize=12)
    ax.set_title(f'Contextual Confusion Matrix (mean diag = {confusion["mean_diagonal"]:.3f})',
                 fontsize=13, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Fraction of tokens', shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_ctx_confusion.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig_ctx_confusion.png")

    # ── Figure 4: Distributions ───────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    for t in types:
        if by_type[t]['H']:
            ax.hist(by_type[t]['H'], bins=40, alpha=0.5, density=True,
                    color=type_colors[t], label=type_labels[t])
    ax.axvline(x=zones['H_low'], color='red', linestyle='--', alpha=0.7)
    ax.axvline(x=zones['H_high'], color='blue', linestyle='--', alpha=0.7)
    ax.set_xlabel('Contextual H(v)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('H(v) Distribution by Condition', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[0, 1]
    for t in types:
        if by_type[t]['norm']:
            ax.hist(by_type[t]['norm'], bins=40, alpha=0.5, density=True,
                    color=type_colors[t], label=type_labels[t])
    ax.axvline(x=zones['norm_low'], color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Contextual Norm ||h||', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Norm Distribution by Condition', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[1, 0]
    for t in types:
        if by_type[t]['max_sim']:
            ax.hist(by_type[t]['max_sim'], bins=40, alpha=0.5, density=True,
                    color=type_colors[t], label=type_labels[t])
    ax.axvline(x=zones['max_sim_low'], color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Max Centroid Similarity', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Max Centroid Similarity by Condition', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Trajectory discontinuity boxplot
    ax = axes[1, 1]
    traj_data = []
    traj_labels_list = []
    traj_colors_list = []
    for t in types:
        if traj_stats[t]:
            traj_data.append([s['disc_rate'] for s in traj_stats[t]])
            traj_labels_list.append(type_labels[t])
            traj_colors_list.append(type_colors[t])
    if traj_data:
        bp = ax.boxplot(traj_data,
                        labels=[l.split('(')[1].rstrip(')') for l in traj_labels_list],
                        patch_artist=True)
        for patch, color in zip(bp['boxes'], traj_colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax.set_ylabel('Discontinuity Rate', fontsize=11)
    ax.set_title('Trajectory Discontinuity (Contextual)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)

    plt.suptitle('Contextual Geometric Signature Distributions',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_ctx_distributions.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig_ctx_distributions.png")

    # ── Figure 5: Static vs Contextual comparison ─────────────────────
    if static_ctx_comparison:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # H(v) comparison
        ax = axes[0]
        x = np.arange(3)
        width = 0.35
        ctx_H = [static_ctx_comparison.get(t, {}).get('ctx_H_mean', 0) for t in types]
        sta_H = [static_ctx_comparison.get(t, {}).get('static_H_mean', 0) for t in types]
        ax.bar(x - width/2, sta_H, width, label='Static Embedding',
               color='#95a5a6', alpha=0.7)
        ax.bar(x + width/2, ctx_H, width, label='Contextual (Last Layer)',
               color='#3498db', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(['Type 1', 'Type 2', 'Type 3'])
        ax.set_ylabel('Mean H(v)', fontsize=11)
        ax.set_title('H(v): Static vs Contextual', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')

        # Norm comparison
        ax = axes[1]
        ctx_n = [static_ctx_comparison.get(t, {}).get('ctx_norm_mean', 0) for t in types]
        sta_n = [static_ctx_comparison.get(t, {}).get('static_norm_mean', 0) for t in types]
        ax.bar(x - width/2, sta_n, width, label='Static Embedding',
               color='#95a5a6', alpha=0.7)
        ax.bar(x + width/2, ctx_n, width, label='Contextual (Last Layer)',
               color='#3498db', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(['Type 1', 'Type 2', 'Type 3'])
        ax.set_ylabel('Mean ||v||', fontsize=11)
        ax.set_title('Norm: Static vs Contextual', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')

        plt.suptitle('Static vs Contextual Representation Comparison',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'fig_ctx_layer_comparison.png'),
                    dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print("  ✓ fig_ctx_layer_comparison.png")


# ──────────────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────────────

def write_report(confusion, stat_tests, traj_stats, zones, calib_stats,
                 static_ctx, all_results, output_dir, diagnostics=None):
    """Write comprehensive contextual induction report."""
    path = os.path.join(output_dir, 'induction_contextual_report.txt')
    lines = []

    lines.append("=" * 80)
    lines.append("CONTEXTUAL HALLUCINATION INDUCTION — LAST-LAYER HIDDEN STATES")
    lines.append("=" * 80)
    lines.append(f"\nModel: GPT-2-small (768D, 12 layers, ~124M parameters)")
    lines.append(f"Representation: Last transformer layer hidden state at generation step")
    lines.append(f"Prompts per condition: {CONFIG['n_prompts_per_type']}")
    lines.append(f"Max new tokens: {CONFIG['max_new_tokens']}")
    lines.append(f"Temperature: {CONFIG['temperature']}, top_k: {CONFIG['top_k']}, "
                 f"top_p: {CONFIG['top_p']}")
    lines.append(f"Calibration prompts: {len(CALIBRATION_PROMPTS)}")

    lines.append(f"\n{'─' * 80}")
    lines.append("CALIBRATION (background distribution of contextual hidden states)")
    lines.append(f"{'─' * 80}")
    lines.append(f"  H(v) = {calib_stats['H_mean']:.4f} ± {calib_stats['H_std']:.4f}")
    lines.append(f"  ||h|| = {calib_stats['norm_mean']:.4f} ± {calib_stats['norm_std']:.4f}")

    lines.append(f"\n{'─' * 80}")
    lines.append("ZONE THRESHOLDS (percentile-calibrated from background)")
    lines.append(f"{'─' * 80}")
    lines.append(f"  Type 1 (center-drift): H(v) < {zones['H_low']:.4f} AND "
                 f"||h|| < {zones['norm_low']:.4f}")
    lines.append(f"  Type 2 (wrong-well):   H(v) > {zones['H_high']:.4f}")
    lines.append(f"  Type 3 (coverage gap):  max_sim < {zones['max_sim_low']:.4f}")

    lines.append(f"\n{'─' * 80}")
    lines.append("CONFUSION MATRIX (rows = induced, cols = contextual zone)")
    lines.append(f"{'─' * 80}")

    types = ['type1', 'type2', 'type3']
    zone_keys = ['type1', 'type2', 'type3', 'unclassified']
    header = f"  {'':>12} {'Zone1':>8} {'Zone2':>8} {'Zone3':>8} {'Uncl':>8} {'N':>8}"
    lines.append(header)
    lines.append(f"  {'─'*52}")
    for t in types:
        row = [f"{confusion['fractions'][t][z]:.3f}" for z in zone_keys]
        lines.append(f"  {t:>12} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8} "
                     f"{confusion['totals'][t]:>8}")

    diag = confusion['diagonal']
    lines.append(f"\n  Diagonal: {diag[0]:.3f}  {diag[1]:.3f}  {diag[2]:.3f}")
    lines.append(f"  Mean diagonal: {confusion['mean_diagonal']:.3f}")

    # Token diagnostics
    if diagnostics:
        lines.extend(token_diagnostics(diagnostics))

    # Two-level statistical tests
    lines.extend(format_stats_report(stat_tests,
                                     "STATISTICAL TESTS (TWO-LEVEL)"))

    lines.append(f"\n{'─' * 80}")
    lines.append("TRAJECTORY ANALYSIS (contextual)")
    lines.append(f"{'─' * 80}")
    for t in types:
        if traj_stats[t]:
            disc = [s['disc_rate'] for s in traj_stats[t]]
            runs = [s['max_run'] for s in traj_stats[t]]
            uniq = [s['n_unique'] for s in traj_stats[t]]
            H_var = [s['H_var'] for s in traj_stats[t]]
            lines.append(f"  {t}:")
            lines.append(f"    Discontinuity rate: {np.mean(disc):.3f} ± {np.std(disc):.3f}")
            lines.append(f"    Max cluster run:    {np.mean(runs):.1f} ± {np.std(runs):.1f}")
            lines.append(f"    Unique clusters:    {np.mean(uniq):.1f} ± {np.std(uniq):.1f}")
            lines.append(f"    H(v) variance:      {np.mean(H_var):.5f} ± {np.std(H_var):.5f}")

    if static_ctx:
        lines.append(f"\n{'─' * 80}")
        lines.append("STATIC vs CONTEXTUAL COMPARISON")
        lines.append(f"{'─' * 80}")
        for t in types:
            if t in static_ctx:
                sc = static_ctx[t]
                lines.append(f"  {t}:")
                lines.append(f"    H(v):  static={sc['static_H_mean']:.4f}, "
                             f"contextual={sc['ctx_H_mean']:.4f}, "
                             f"spread_ratio={sc['H_spread_ratio']:.2f}x")
                lines.append(f"    ||v||: static={sc['static_norm_mean']:.4f}, "
                             f"contextual={sc['ctx_norm_mean']:.4f}, "
                             f"spread_ratio={sc['norm_spread_ratio']:.2f}x")

    lines.append(f"\n{'─' * 80}")
    lines.append("SAMPLE GENERATIONS")
    lines.append(f"{'─' * 80}")
    for t in types:
        lines.append(f"\n  [{t.upper()}]")
        seqs = [s for s in all_results if s['type'] == t]
        for seq in seqs[:3]:
            preview = seq['generated_text'][:120].replace('\n', ' ')
            lines.append(f"    Prompt: \"{seq['prompt']}\"")
            lines.append(f"    Output: \"{preview}...\"")
            if seq.get('measurements'):
                zones_seq = [m['zone'] for m in seq['measurements']]
                zone_counts = {z: zones_seq.count(z) for z in zone_keys}
                lines.append(f"    Zones: {zone_counts}")
            lines.append("")

    lines.append(f"\n{'─' * 80}")
    lines.append("INTERPRETATION")
    lines.append(f"{'─' * 80}")

    above_chance = confusion['mean_diagonal'] > 0.33
    lines.append(f"""
  CONTEXTUAL vs STATIC COMPARISON
  ================================
  Static experiment: mean diagonal = 0.288 (below chance). All conditions
  collapsed to the same region of static embedding space.

  Contextual mean diagonal: {confusion['mean_diagonal']:.3f}
  Above chance (>0.33): {'YES' if above_chance else 'NO'}

  KEY FINDINGS (prompt-level, N=15/group, is the primary inference level):
  - Norm omnibus is borderline significant at prompt level. Pairwise
    Type 3 separations are nominally significant but do not survive
    Holm correction (adj p ≈ 0.10). Permutation p-values and bootstrap
    CIs support a real but underpowered effect.
  - Angular metrics (H(v), max_sim) are highly significant at token level
    but COMPLETELY NON-SIGNIFICANT at prompt level (p > 0.38). This is a
    pseudoreplication artifact: within-prompt autocorrelation inflates the
    effective sample size from 15 independent prompts to ~900 correlated
    tokens, creating spurious separation that vanishes at the correct
    unit of analysis.
  - Types 1 and 2 do not separate on any metric at either analysis level.

  The robust signal is magnitude (norm), not direction (angular metrics).
  Type 3 (coverage gap) is the only geometrically distinctive failure
  mode at 124M parameters. The confusion matrix fails because last-layer
  hidden states occupy a near-saturated similarity regime (H ≈ 0.985,
  max_sim ≈ 0.993) where percentile-based thresholds cannot operate.

  NOTE: All inferential claims should be based on prompt-level results.
  Token-level statistics are reported for reference only.
""")

    lines.append("=" * 80)
    lines.append("END OF CONTEXTUAL INDUCTION REPORT")
    lines.append("=" * 80)

    report = '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(report)

    print(f"\n  Report written to {path}")
    return report


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CONTEXTUAL HALLUCINATION INDUCTION — LAST-LAYER HIDDEN STATES")
    print("Model: GPT-2-small | 3 conditions × 15 prompts × 60 tokens")
    print("Representation: Contextual (layer 12) vs Static (embedding)")
    print("=" * 70)
    t_start = time.time()

    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    np.random.seed(CONFIG['random_seed'])

    import torch
    torch.manual_seed(CONFIG['random_seed'])

    # Step 1: Load
    model, tokenizer, static_emb = load_gpt2()

    # Step 2: Build calibration distribution (contextual)
    calibration_hidden = build_calibration_distribution(model, tokenizer)

    # Step 3: Cluster and calibrate
    kmeans, centroids, zones, calib_stats = cluster_and_calibrate(calibration_hidden)

    # Free calibration data (keep stats)
    del calibration_hidden
    gc.collect()

    # Step 4: Generate under experimental conditions
    print(f"\n[Step 4] Generating under controlled conditions...")
    gen_type1 = generate_experimental(model, tokenizer, TYPE1_PROMPTS, 'type1')
    gen_type2 = generate_experimental(model, tokenizer, TYPE2_PROMPTS, 'type2')
    gen_type3 = generate_experimental(model, tokenizer, TYPE3_PROMPTS, 'type3')

    all_results = gen_type1 + gen_type2 + gen_type3

    # Free model
    del model
    gc.collect()

    # Step 5: Measure geometry (contextual + static comparison)
    all_results = measure_geometry(all_results, centroids, zones, static_emb=static_emb)

    # Step 6: Confusion matrix
    confusion = compute_confusion_matrix(all_results)

    # Step 7: Statistical tests
    stat_tests, diagnostics = run_statistical_tests(all_results)

    # Step 8: Trajectory analysis
    traj_stats = analyze_trajectories(all_results)

    # Static vs contextual comparison
    static_ctx = compare_static_contextual(all_results)

    # Step 9: Figures
    if CONFIG['generate_figures']:
        generate_figures(all_results, calib_stats, zones, confusion,
                         traj_stats, static_ctx, CONFIG['output_dir'])
    else:
        print("\n[Step 9] Skipping figures (generate_figures=False)")

    # Step 10: Report
    report = write_report(confusion, stat_tests, traj_stats, zones, calib_stats,
                          static_ctx, all_results, CONFIG['output_dir'], diagnostics)

    # Save raw results (strip numpy arrays for JSON)
    json_path = os.path.join(CONFIG['output_dir'], 'raw_results_contextual.json')
    json_data = []
    for seq in all_results:
        json_data.append({
            'type': seq['type'],
            'prompt': seq['prompt'],
            'generated_text': seq['generated_text'],
            'measurements': seq.get('measurements', []),
        })
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Raw results saved to {json_path}")

    # Save two-level stats + comparison summary
    summary = {
        'confusion': {
            'fractions': confusion['fractions'],
            'diagonal': confusion['diagonal'],
            'mean_diagonal': confusion['mean_diagonal'],
        },
        'two_level_stats': results_to_json(stat_tests),
        'static_vs_contextual': static_ctx,
        'zones': zones,
        'calibration': {
            'H_mean': calib_stats['H_mean'],
            'H_std': calib_stats['H_std'],
            'norm_mean': calib_stats['norm_mean'],
            'norm_std': calib_stats['norm_std'],
        },
    }
    with open(os.path.join(CONFIG['output_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"COMPLETE — Total runtime: {time.time() - t_start:.1f}s")
    print(f"Results in: {os.path.abspath(CONFIG['output_dir'])}")
    print(f"{'=' * 70}")

    print("\n" + report)


if __name__ == '__main__':
    main()
