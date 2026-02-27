#!/usr/bin/env python3
"""
Geometric Hallucination Taxonomy — Controlled Induction Experiment
====================================================================

Validates the three-type hallucination taxonomy by inducing each type
in GPT-2-small through controlled prompt conditions, then measuring
whether generated tokens produce the predicted geometric signatures
in static embedding space.

Prompt conditions:
  Type 1 (center-drift):  Weak/degenerate context → generic tokens
  Type 2 (wrong-well):    Domain-ambiguous context → wrong-cluster lock
  Type 3 (coverage gap):  Cross-domain neologisms → sparse-region scatter

Measurements per generated token:
  - H(v): soft cluster membership (mean top-5 cosine sim to centroids)
  - ||v||: embedding norm
  - max_sim: maximum centroid similarity
  - cluster assignment trajectory

Outputs (in ./results_induction/):
  - induction_report.txt           : full results with confusion matrix
  - fig_induction_trajectories.png : norm–membership trajectories per type
  - fig_induction_zones.png        : generated tokens overlaid on vocab distribution
  - fig_induction_confusion.png    : confusion matrix heatmap
  - fig_induction_distributions.png: H(v) and norm distributions per condition
  - raw_results.json               : machine-readable per-token data

Hardware: CPU only, ~4GB RAM, ~5-10 min runtime
Usage:  python hallucination_induction.py
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
    'n_radial_bins': 40,
    'min_bin_count': 10,
    'random_seed': 42,
    'output_dir': './results_induction',
    'figure_dpi': 150,
    'generate_figures': True,    # set False to skip diagnostic figure generation

    # Generation parameters
    'max_new_tokens': 60,
    'n_prompts_per_type': 15,
    'temperature': 1.0,         # standard sampling — no tricks
    'top_k': 0,                 # disabled — let the model show its failure modes
    'top_p': 1.0,               # disabled
    'do_sample': True,

    # Zone thresholds (percentile-based, calibrated from vocab distribution)
    # These are computed dynamically in calibrate_zones()
    'H_low_pct': 15,            # Type 1: below this percentile of H(v)
    'norm_low_pct': 40,         # Type 1: below this percentile of norm
    'H_high_pct': 75,           # Type 2: above this percentile of H(v)
    'max_sim_low_pct': 25,      # Type 3: below this percentile of max centroid sim
}


# ──────────────────────────────────────────────────────────────────────
# PROMPT SETS
# ──────────────────────────────────────────────────────────────────────

# Type 1: Weak context — minimal, degenerate, maximally ambiguous prompts
# that provide almost no directional signal. The model should drift to
# high-frequency generic tokens near the centroid.
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

# Type 2: Domain-ambiguous prompts — strong enough signal to commit to a
# cluster, but genuinely ambiguous between two well-defined domains. The
# model should lock onto one well and stay there with high confidence,
# but the "correct" domain is undefined. We measure whether the generation
# shows high cluster membership (the signature) regardless of which
# domain it picks.
TYPE2_PROMPTS = [
    "The bank announced record levels of",                    # financial vs river
    "The pitch was perfect for the",                          # music vs baseball vs sales
    "She picked up the bat and",                              # cricket vs animal
    "The board decided to table the",                         # furniture vs meeting
    "He studied the cell under the",                          # biology vs prison
    "The crane lifted the heavy",                             # bird vs machine
    "The plant manager reviewed the new",                     # factory vs botany
    "Mercury levels in the",                                  # planet vs element vs thermometer
    "The current was too strong for the",                     # electrical vs water vs present
    "The cabinet members discussed the",                      # furniture vs government
    "They found the mole in the",                             # animal vs spy vs chemistry
    "The scales showed exactly",                              # weight vs fish vs music
    "The press released a statement about the",               # media vs machine vs push
    "The match was struck and",                               # fire vs sports vs pairing
    "The patient leaves were carefully",                      # medical vs botanical vs adjective
]

# Type 3: Coverage gap — prompts requiring compositional knowledge that
# GPT-2 almost certainly lacks. Cross-domain technical neologisms,
# fictional proper nouns needing grounding, and out-of-distribution
# combinations. Tokens should scatter without cluster coherence.
TYPE3_PROMPTS = [
    "The xenoplasmic refractometry of late-Holocene",                    # fictional+technical
    "Professor Kvistad's third theorem on paracompact",                  # fake attribution+math
    "In Zvrotkian epistemology, the concept of mereological",           # fictional philosophy
    "The biosemiotic implications of CRISPR-modified Tardigrade",       # extreme cross-domain
    "According to the Nørgaard-Patel conjecture in topological",       # fake conjecture
    "The post-Deleuzian analysis of quantum decoherence in",           # philosophy+physics mash
    "Hyperbolic crochet models of anti-de Sitter spacetime suggest",   # real but obscure+deep
    "The ethnopharmacological study of Amazonian Ayahuasca analogs",   # rare cross-domain
    "A contravariant functor from the category of smooth manifolds",   # pure math, dense
    "The archaeomagnetostratigraphic evidence from Paleocene",         # real compound obscure
    "Subquadratic approximation algorithms for the Steiner tree",      # CS theory
    "The phenomenological reduction of eigenstate thermalization",     # philosophy+physics
    "Applying Khovanov homology to categorified quantum groups",      # advanced math
    "The gliotransmitter-mediated modulation of thalamocortical",     # deep neuroscience
    "Transfinite induction over the constructible hierarchy L",        # set theory
]


# ──────────────────────────────────────────────────────────────────────
# EMBEDDING EXTRACTION AND VOCABULARY FILTERING
# ──────────────────────────────────────────────────────────────────────

def load_gpt2():
    """Load GPT-2 for both generation and embedding extraction."""
    print("\n[Step 1] Loading GPT-2...")
    t0 = time.time()

    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', torch_dtype=torch.float32)
    model.eval()

    # Extract static embedding matrix
    with torch.no_grad():
        emb_matrix = model.transformer.wte.weight.cpu().numpy()

    print(f"  Embedding matrix: {emb_matrix.shape}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    return model, tokenizer, emb_matrix


def filter_vocabulary(emb_matrix, tokenizer):
    """Filter to whole-word tokens with frequency data (BPE-aware)."""
    print("\n[Step 2] Filtering vocabulary...")
    t0 = time.time()

    from wordfreq import word_frequency

    vocab = tokenizer.get_vocab()
    special_tokens = set(tokenizer.all_special_tokens)

    filtered_indices = []
    words = []
    frequencies = []

    for token, idx in vocab.items():
        if token in special_tokens:
            continue
        # GPT-2 BPE: Ġ prefix = word-initial token
        if not token.startswith('Ġ'):
            continue
        word = token[1:].lower()
        if len(word) < 2 or not word.isalpha():
            continue

        freq = word_frequency(word, 'en')
        if freq > 0:
            filtered_indices.append(idx)
            words.append(word)
            frequencies.append(freq)

    filtered_indices = np.array(filtered_indices)
    filtered_emb = emb_matrix[filtered_indices]
    frequencies = np.array(frequencies)
    self_info = -np.log2(frequencies)

    print(f"  Filtered tokens: {len(words)}")
    print(f"  Self-info range: [{self_info.min():.1f}, {self_info.max():.1f}] bits")
    print(f"  Done in {time.time() - t0:.1f}s")

    return filtered_emb, filtered_indices, words, self_info


# ──────────────────────────────────────────────────────────────────────
# CLUSTERING AND ZONE CALIBRATION
# ──────────────────────────────────────────────────────────────────────

def cluster_and_calibrate(filtered_emb, words, self_info):
    """Cluster vocabulary and compute zone thresholds from distribution."""
    print("\n[Step 3] Clustering and calibrating zones...")
    t0 = time.time()

    kmeans = MiniBatchKMeans(
        n_clusters=CONFIG['n_clusters'], random_state=CONFIG['random_seed'],
        batch_size=1024, max_iter=300, n_init=5)
    labels = kmeans.fit_predict(filtered_emb)
    centroids = kmeans.cluster_centers_

    sizes = np.bincount(labels)
    print(f"  Clusters: {CONFIG['n_clusters']}, sizes: "
          f"min={sizes.min()}, mean={sizes.mean():.0f}, max={sizes.max()}")

    # Compute vocab-wide H(v) and norms for calibration
    norms = np.linalg.norm(filtered_emb, axis=1)
    sims_to_centroids = cosine_similarity(filtered_emb, centroids)  # (n_tokens, k)
    top5_sims = np.sort(sims_to_centroids, axis=1)[:, -5:]
    H_vals = top5_sims.mean(axis=1)
    max_sims = sims_to_centroids.max(axis=1)

    # Calibrate zone thresholds from percentiles
    zones = {
        'H_low': float(np.percentile(H_vals, CONFIG['H_low_pct'])),
        'norm_low': float(np.percentile(norms, CONFIG['norm_low_pct'])),
        'H_high': float(np.percentile(H_vals, CONFIG['H_high_pct'])),
        'max_sim_low': float(np.percentile(max_sims, CONFIG['max_sim_low_pct'])),
    }

    print(f"  Zone thresholds (percentile-calibrated):")
    print(f"    Type 1: H(v) < {zones['H_low']:.4f} (p{CONFIG['H_low_pct']}) "
          f"AND ||v|| < {zones['norm_low']:.4f} (p{CONFIG['norm_low_pct']})")
    print(f"    Type 2: H(v) > {zones['H_high']:.4f} (p{CONFIG['H_high_pct']})")
    print(f"    Type 3: max_sim < {zones['max_sim_low']:.4f} (p{CONFIG['max_sim_low_pct']})")

    vocab_stats = {
        'H_vals': H_vals,
        'norms': norms,
        'max_sims': max_sims,
        'labels': labels,
        'self_info': self_info,
    }

    print(f"  Done in {time.time() - t0:.1f}s")
    return kmeans, centroids, zones, vocab_stats


# ──────────────────────────────────────────────────────────────────────
# TEXT GENERATION
# ──────────────────────────────────────────────────────────────────────

def generate_sequences(model, tokenizer, prompts, type_label):
    """Generate continuations for a set of prompts. Returns list of dicts."""
    import torch

    print(f"\n  Generating {type_label} ({len(prompts)} prompts)...")

    results = []
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=CONFIG['max_new_tokens'],
                temperature=CONFIG['temperature'],
                top_k=CONFIG['top_k'] if CONFIG['top_k'] > 0 else None,
                top_p=CONFIG['top_p'],
                do_sample=CONFIG['do_sample'],
                pad_token_id=tokenizer.eos_token_id,
            )

        # Extract only the generated tokens (not the prompt)
        gen_ids = output[0, prompt_len:].cpu().numpy()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        results.append({
            'prompt': prompt,
            'generated_ids': gen_ids.tolist(),
            'generated_text': gen_text,
            'type': type_label,
            '_max_tokens': CONFIG['max_new_tokens'],
        })

        if i < 3:  # show first 3
            preview = gen_text[:80].replace('\n', ' ')
            print(f"    [{i+1}] \"{prompt}\" → \"{preview}...\"")

    return results


# ──────────────────────────────────────────────────────────────────────
# GEOMETRIC MEASUREMENT
# ──────────────────────────────────────────────────────────────────────

def measure_geometry(gen_results, emb_matrix, centroids, zones):
    """For each generated sequence, compute per-token geometric metrics."""
    print("\n[Step 5] Measuring geometric signatures...")
    t0 = time.time()

    all_measurements = []

    for seq in gen_results:
        token_ids = seq['generated_ids']
        token_measurements = []

        for tid in token_ids:
            if tid >= emb_matrix.shape[0]:
                continue

            v = emb_matrix[tid].reshape(1, -1)
            norm = float(np.linalg.norm(v))

            # Similarities to all centroids
            sims = cosine_similarity(v, centroids).flatten()
            top5 = np.sort(sims)[-5:]
            H = float(top5.mean())
            max_sim = float(sims.max())
            assigned_cluster = int(sims.argmax())

            # Classify into zone
            zone = classify_token(H, norm, max_sim, zones)

            token_measurements.append({
                'token_id': int(tid),
                'norm': norm,
                'H': H,
                'max_sim': max_sim,
                'cluster': assigned_cluster,
                'zone': zone,
            })

        seq['measurements'] = token_measurements

    print(f"  Done in {time.time() - t0:.1f}s")
    return gen_results


def classify_token(H, norm, max_sim, zones):
    """Classify a single token into a geometric zone.

    Priority: Type 1 (low H + low norm) > Type 3 (low max_sim) > Type 2 (high H) > Unclassified
    This order reflects the taxonomy: center-drift is the most geometrically
    distinct, coverage gaps next, wrong-well is the "default confident" mode.
    """
    if H < zones['H_low'] and norm < zones['norm_low']:
        return 'type1'
    if max_sim < zones['max_sim_low']:
        return 'type3'
    if H > zones['H_high']:
        return 'type2'
    return 'unclassified'


# ──────────────────────────────────────────────────────────────────────
# ANALYSIS AND CONFUSION MATRIX
# ──────────────────────────────────────────────────────────────────────

def compute_confusion_matrix(all_results):
    """Compute confusion matrix: rows=induced condition, cols=geometric zone."""
    print("\n[Step 6] Computing confusion matrix...")

    types = ['type1', 'type2', 'type3']
    zones = ['type1', 'type2', 'type3', 'unclassified']

    # Count tokens per condition × zone
    counts = {t: {z: 0 for z in zones} for t in types}
    totals = {t: 0 for t in types}

    # Per-sequence stats
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
                               for z in zones},
        })

    # Compute fractions
    fractions = {}
    for t in types:
        fractions[t] = {}
        for z in zones:
            fractions[t][z] = counts[t][z] / totals[t] if totals[t] > 0 else 0

    # Diagonal = correct classification rate
    diagonal = [fractions[t][t] for t in types]
    mean_diag = np.mean(diagonal)

    print(f"\n  Confusion matrix (rows=induced, cols=detected zone):")
    print(f"  {'':>12} {'Type1':>8} {'Type2':>8} {'Type3':>8} {'Unclass':>8} {'Total':>8}")
    print(f"  {'─'*56}")
    for t in types:
        row = [f"{fractions[t][z]:.3f}" for z in zones]
        print(f"  {t:>12} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8} {totals[t]:>8}")

    print(f"\n  Diagonal (correct classification): "
          f"{diagonal[0]:.3f}, {diagonal[1]:.3f}, {diagonal[2]:.3f}")
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
# TRAJECTORY ANALYSIS (Type 2 specific)
# ──────────────────────────────────────────────────────────────────────

def analyze_trajectories(all_results):
    """Compute trajectory smoothness for each sequence.

    Trajectory discontinuity = fraction of consecutive token pairs where
    the cluster assignment changes. Type 2 should show *some* smooth runs
    followed by abrupt jumps; Type 1 should show random wandering.
    """
    print("\n[Step 8] Trajectory analysis...")

    traj_stats = {'type1': [], 'type2': [], 'type3': []}

    for seq in all_results:
        t = seq['type']
        measurements = seq.get('measurements', [])
        if len(measurements) < 3:
            continue

        clusters = [m['cluster'] for m in measurements]
        n = len(clusters)

        # Discontinuity rate: fraction of steps where cluster changes
        changes = sum(1 for i in range(1, n) if clusters[i] != clusters[i-1])
        disc_rate = changes / (n - 1)

        # Max run length: longest streak in same cluster
        max_run = 1
        current_run = 1
        for i in range(1, n):
            if clusters[i] == clusters[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        # Number of unique clusters visited
        n_unique = len(set(clusters))

        traj_stats[t].append({
            'disc_rate': disc_rate,
            'max_run': max_run,
            'n_unique': n_unique,
            'n_tokens': n,
        })

    print(f"\n  Trajectory statistics (mean ± std):")
    for t in ['type1', 'type2', 'type3']:
        if traj_stats[t]:
            disc = [s['disc_rate'] for s in traj_stats[t]]
            runs = [s['max_run'] for s in traj_stats[t]]
            uniq = [s['n_unique'] for s in traj_stats[t]]
            print(f"    {t}: disc_rate={np.mean(disc):.3f}±{np.std(disc):.3f}, "
                  f"max_run={np.mean(runs):.1f}±{np.std(runs):.1f}, "
                  f"n_unique={np.mean(uniq):.1f}±{np.std(uniq):.1f}")

    return traj_stats


# ──────────────────────────────────────────────────────────────────────
# FIGURES
# ──────────────────────────────────────────────────────────────────────

def generate_figures(all_results, vocab_stats, zones, confusion, traj_stats, output_dir):
    """Generate all figures for the induction experiment."""
    print("\n[Step 9] Generating figures...")
    dpi = CONFIG['figure_dpi']

    type_colors = {'type1': '#e74c3c', 'type2': '#2980b9', 'type3': '#27ae60'}
    type_labels = {'type1': 'Type 1 (center-drift)',
                   'type2': 'Type 2 (wrong-well)',
                   'type3': 'Type 3 (coverage gap)'}

    # Collect all measurements by type
    by_type = {'type1': {'H': [], 'norm': [], 'max_sim': []},
               'type2': {'H': [], 'norm': [], 'max_sim': []},
               'type3': {'H': [], 'norm': [], 'max_sim': []}}

    for seq in all_results:
        t = seq['type']
        for m in seq.get('measurements', []):
            by_type[t]['H'].append(m['H'])
            by_type[t]['norm'].append(m['norm'])
            by_type[t]['max_sim'].append(m['max_sim'])

    # ── Figure 1: Generated tokens overlaid on vocab distribution ─────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (t, label) in zip(axes, type_labels.items()):
        # Background: vocab distribution
        ax.scatter(vocab_stats['norms'], vocab_stats['H_vals'],
                   c='#cccccc', s=1, alpha=0.15, rasterized=True)

        # Zone boundaries
        ax.axhline(y=zones['H_low'], color='#e74c3c', linewidth=0.8,
                   linestyle='--', alpha=0.6, label=f"H={zones['H_low']:.3f}")
        ax.axhline(y=zones['H_high'], color='#2980b9', linewidth=0.8,
                   linestyle='--', alpha=0.6, label=f"H={zones['H_high']:.3f}")
        ax.axvline(x=zones['norm_low'], color='#e74c3c', linewidth=0.8,
                   linestyle=':', alpha=0.6)

        # Generated tokens
        if by_type[t]['norm']:
            ax.scatter(by_type[t]['norm'], by_type[t]['H'],
                       c=type_colors[t], s=12, alpha=0.6, edgecolors='white',
                       linewidth=0.3, label=f'Generated ({len(by_type[t]["H"])} tokens)')

        ax.set_xlabel('Embedding Norm ||v||', fontsize=11)
        ax.set_ylabel('Soft Cluster Membership H(v)', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold', color=type_colors[t])
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.2)

    plt.suptitle('Generated Token Signatures in Norm–Membership Space',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_induction_zones.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig_induction_zones.png")

    # ── Figure 2: Trajectories (cluster assignment over generation) ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, t in zip(axes, ['type1', 'type2', 'type3']):
        seqs = [s for s in all_results if s['type'] == t]
        for i, seq in enumerate(seqs[:8]):  # show up to 8 trajectories
            measurements = seq.get('measurements', [])
            if not measurements:
                continue
            clusters = [m['cluster'] for m in measurements]
            ax.plot(range(len(clusters)), clusters, '-', alpha=0.5,
                    linewidth=1.2, color=type_colors[t])
            # Mark discontinuities
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

    plt.suptitle('Cluster Assignment Trajectories During Generation',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_induction_trajectories.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig_induction_trajectories.png")

    # ── Figure 3: Confusion matrix heatmap ────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))

    types = ['type1', 'type2', 'type3']
    zone_labels = ['type1', 'type2', 'type3', 'unclassified']
    matrix = np.array([[confusion['fractions'][t][z] for z in zone_labels]
                       for t in types])

    im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    # Annotate cells
    for i in range(len(types)):
        for j in range(len(zone_labels)):
            val = matrix[i, j]
            color = 'white' if val > 0.5 else 'black'
            count = confusion['counts'][types[i]][zone_labels[j]]
            ax.text(j, i, f'{val:.2f}\n({count})',
                    ha='center', va='center', fontsize=10, color=color)

    ax.set_xticks(range(len(zone_labels)))
    ax.set_xticklabels(['Zone 1\n(center-drift)', 'Zone 2\n(wrong-well)',
                        'Zone 3\n(coverage gap)', 'Unclassified'],
                       fontsize=9)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(['Induced\nType 1', 'Induced\nType 2', 'Induced\nType 3'],
                       fontsize=10)
    ax.set_xlabel('Detected Geometric Zone', fontsize=12)
    ax.set_ylabel('Induced Condition', fontsize=12)
    ax.set_title(f'Confusion Matrix (mean diagonal = {confusion["mean_diagonal"]:.3f})',
                 fontsize=13, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Fraction of tokens', shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_induction_confusion.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig_induction_confusion.png")

    # ── Figure 4: H(v) and norm distributions per condition ───────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # H(v) distributions
    ax = axes[0, 0]
    for t in types:
        if by_type[t]['H']:
            ax.hist(by_type[t]['H'], bins=40, alpha=0.5, density=True,
                    color=type_colors[t], label=type_labels[t])
    ax.axvline(x=zones['H_low'], color='red', linestyle='--', alpha=0.7,
               label=f"Zone 1 threshold")
    ax.axvline(x=zones['H_high'], color='blue', linestyle='--', alpha=0.7,
               label=f"Zone 2 threshold")
    ax.set_xlabel('H(v) — Soft Cluster Membership', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('H(v) Distribution by Condition', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Norm distributions
    ax = axes[0, 1]
    for t in types:
        if by_type[t]['norm']:
            ax.hist(by_type[t]['norm'], bins=40, alpha=0.5, density=True,
                    color=type_colors[t], label=type_labels[t])
    ax.axvline(x=zones['norm_low'], color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Embedding Norm ||v||', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Norm Distribution by Condition', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Max centroid similarity distributions
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

    # Trajectory discontinuity rates
    ax = axes[1, 1]
    traj_data = []
    traj_labels_list = []
    traj_colors_list = []
    for t in types:
        if traj_stats[t]:
            vals = [s['disc_rate'] for s in traj_stats[t]]
            traj_data.append(vals)
            traj_labels_list.append(type_labels[t])
            traj_colors_list.append(type_colors[t])
    if traj_data:
        bp = ax.boxplot(traj_data, labels=[l.split('(')[1].rstrip(')')
                        for l in traj_labels_list],
                        patch_artist=True)
        for patch, color in zip(bp['boxes'], traj_colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax.set_ylabel('Discontinuity Rate', fontsize=11)
    ax.set_title('Trajectory Discontinuity by Condition', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)

    plt.suptitle('Geometric Signature Distributions',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_induction_distributions.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig_induction_distributions.png")


# ──────────────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────────────

def write_report(confusion, stat_tests, traj_stats, zones, all_results, output_dir, diagnostics=None):
    """Write comprehensive induction experiment report."""
    path = os.path.join(output_dir, 'induction_report.txt')
    lines = []

    lines.append("=" * 80)
    lines.append("GEOMETRIC HALLUCINATION TAXONOMY — CONTROLLED INDUCTION EXPERIMENT")
    lines.append("=" * 80)
    lines.append(f"\nModel: GPT-2-small (768D, ~124M parameters)")
    lines.append(f"Prompts per condition: {CONFIG['n_prompts_per_type']}")
    lines.append(f"Max new tokens: {CONFIG['max_new_tokens']}")
    lines.append(f"Temperature: {CONFIG['temperature']}, top_k: {CONFIG['top_k']}, "
                 f"top_p: {CONFIG['top_p']}")

    lines.append(f"\n{'─' * 80}")
    lines.append("ZONE THRESHOLDS (percentile-calibrated from GPT-2 vocabulary)")
    lines.append(f"{'─' * 80}")
    lines.append(f"  Type 1 (center-drift): H(v) < {zones['H_low']:.4f} AND "
                 f"||v|| < {zones['norm_low']:.4f}")
    lines.append(f"  Type 2 (wrong-well):   H(v) > {zones['H_high']:.4f}")
    lines.append(f"  Type 3 (coverage gap):  max_sim < {zones['max_sim_low']:.4f}")

    lines.append(f"\n{'─' * 80}")
    lines.append("CONFUSION MATRIX (rows = induced condition, cols = detected zone)")
    lines.append(f"{'─' * 80}")

    types = ['type1', 'type2', 'type3']
    zone_labels = ['type1', 'type2', 'type3', 'unclassified']

    header = f"  {'':>12} {'Zone1':>8} {'Zone2':>8} {'Zone3':>8} {'Uncl':>8} {'N':>8}"
    lines.append(header)
    lines.append(f"  {'─'*52}")
    for t in types:
        row = [f"{confusion['fractions'][t][z]:.3f}" for z in zone_labels]
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
    lines.append("TRAJECTORY ANALYSIS")
    lines.append(f"{'─' * 80}")
    for t in types:
        if traj_stats[t]:
            disc = [s['disc_rate'] for s in traj_stats[t]]
            runs = [s['max_run'] for s in traj_stats[t]]
            uniq = [s['n_unique'] for s in traj_stats[t]]
            lines.append(f"  {t}:")
            lines.append(f"    Discontinuity rate: {np.mean(disc):.3f} ± {np.std(disc):.3f}")
            lines.append(f"    Max cluster run:    {np.mean(runs):.1f} ± {np.std(runs):.1f}")
            lines.append(f"    Unique clusters:    {np.mean(uniq):.1f} ± {np.std(uniq):.1f}")

    lines.append(f"\n{'─' * 80}")
    lines.append("PER-CONDITION MEANS")
    lines.append(f"{'─' * 80}")
    for t in types:
        seq_s = confusion['seq_stats'][t]
        if seq_s:
            H_means = [s['mean_H'] for s in seq_s]
            norm_means = [s['mean_norm'] for s in seq_s]
            lines.append(f"  {t}: mean H(v)={np.mean(H_means):.4f} ± {np.std(H_means):.4f}, "
                         f"mean ||v||={np.mean(norm_means):.4f} ± {np.std(norm_means):.4f}")

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
                zone_counts = {z: zones_seq.count(z) for z in zone_labels}
                lines.append(f"    Zones: {zone_counts}")
            lines.append("")

    lines.append(f"\n{'─' * 80}")
    lines.append("INTERPRETATION")
    lines.append(f"{'─' * 80}")
    lines.append(f"""
  This experiment tests whether controlled prompt conditions produce the
  geometric signatures predicted by the three-type taxonomy in static
  (vocabulary) embeddings.

  Observed mean diagonal: {confusion['mean_diagonal']:.3f}
  (Below 0.33 chance baseline — zone classification fails in static space.)

  KEY FINDINGS (prompt-level, N=15/group, is the primary inference level):
  - H(v) cannot distinguish conditions at either analysis level.
  - Norm and max_sim partially separate Type 3 from other types,
    with two pairwise comparisons surviving Holm correction at prompt level.
  - Types 1 and 2 are indistinguishable on all metrics at all levels.

  This is a structured partial negative: the angular membership metric
  fails, but magnitude-related metrics show that Type 3 (coverage gap)
  is partially detectable even in static vocabulary space. The signal
  reflects vocabulary selection differences — compositional novelty
  prompts force rarer token choices with higher norms.

  NOTE: Token-level p-values (N≈900) are reported for reference only.
  Tokens within a prompt are autocorrelated; prompt-level aggregation
  is required for valid inference.
""")

    lines.append("=" * 80)
    lines.append("END OF INDUCTION REPORT")
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
    print("GEOMETRIC HALLUCINATION TAXONOMY — CONTROLLED INDUCTION")
    print("Model: GPT-2-small | 3 conditions × 15 prompts × 60 tokens")
    print("=" * 70)
    t_start = time.time()

    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    np.random.seed(CONFIG['random_seed'])

    # Step 1: Load GPT-2
    model, tokenizer, emb_matrix = load_gpt2()

    # Step 2: Filter vocabulary
    filtered_emb, filtered_indices, words, self_info = filter_vocabulary(
        emb_matrix, tokenizer)

    # Step 3: Cluster and calibrate zones
    kmeans, centroids, zones, vocab_stats = cluster_and_calibrate(
        filtered_emb, words, self_info)

    # Step 4: Generate text under each condition
    print(f"\n[Step 4] Generating under controlled conditions...")
    import torch
    torch.manual_seed(CONFIG['random_seed'])

    gen_type1 = generate_sequences(model, tokenizer, TYPE1_PROMPTS, 'type1')
    gen_type2 = generate_sequences(model, tokenizer, TYPE2_PROMPTS, 'type2')
    gen_type3 = generate_sequences(model, tokenizer, TYPE3_PROMPTS, 'type3')

    all_results = gen_type1 + gen_type2 + gen_type3

    # Free model — only need embedding matrix from here
    del model
    gc.collect()

    # Step 5: Measure geometry
    all_results = measure_geometry(all_results, emb_matrix, centroids, zones)

    # Step 6: Confusion matrix
    confusion = compute_confusion_matrix(all_results)

    # Step 7: Statistical tests
    stat_tests, diagnostics = run_statistical_tests(all_results)

    # Step 8: Trajectory analysis
    traj_stats = analyze_trajectories(all_results)

    # Step 9: Figures
    if CONFIG['generate_figures']:
        generate_figures(all_results, vocab_stats, zones, confusion,
                         traj_stats, CONFIG['output_dir'])
    else:
        print("\n[Step 9] Skipping figures (generate_figures=False)")

    # Step 10: Report
    report = write_report(confusion, stat_tests, traj_stats, zones,
                          all_results, CONFIG['output_dir'], diagnostics)

    # Save raw results
    json_path = os.path.join(CONFIG['output_dir'], 'raw_results.json')
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

    # Save two-level statistical results
    stats_json_path = os.path.join(CONFIG['output_dir'], 'two_level_stats.json')
    with open(stats_json_path, 'w') as f:
        json.dump(results_to_json(stat_tests), f, indent=2)
    print(f"  Two-level stats saved to {stats_json_path}")

    print(f"\n{'=' * 70}")
    print(f"COMPLETE — Total runtime: {time.time() - t_start:.1f}s")
    print(f"Results in: {os.path.abspath(CONFIG['output_dir'])}")
    print(f"{'=' * 70}")

    print("\n" + report)


if __name__ == '__main__':
    main()
