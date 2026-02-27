# Validating a Geometric Hallucination Taxonomy Through Controlled Induction

Code and data for the paper *"From Prerequisites to Predictions: Validating a Geometric Hallucination Taxonomy Through Controlled Induction"*.

We test whether a geometric hallucination taxonomy — classifying failures as center-drift (Type 1), wrong-well convergence (Type 2), or coverage gaps (Type 3) — produces measurable signatures in GPT-2's representations when each failure type is deliberately triggered through controlled prompt design.

- **Type 1 (Center-drift)** — minimal prompts ("The", "It is") produce generic high-frequency continuations
- **Type 2 (Wrong-well)** — ambiguous prompts ("The bank announced record levels of") force commitment to one domain
- **Type 3 (Coverage gap)** — compositionally novel prompts ("The xenoplasmic refractometry of late-Holocene") encounter no training support

Key findings across 20 independent generation runs (K=20 stability protocol):

| Finding | Static Embeddings | Contextual Hidden States |
|---------|-------------------|--------------------------|
| Type 3 norm direction | 20/20 highest | 19/20 lowest |
| Type 3 norm sig/20 (Holm) | 18/20 (14/20) | 4/20 (2/20) |
| Type 1/2 separation | ≤2/20 | ≤2/20 |
| Pseudoreplication inflation | — | 4–16× across all comparisons |

## Repository Structure

```
├── hallucination_induction.py             # Experiment 1: static embeddings
├── hallucination_induction_contextual.py  # Experiment 2: contextual hidden states
├── hallucination_stats.py                 # Shared two-level statistical analysis module
├── run_multirun.py                        # K-run stability wrapper (main entry point)
├── generate_figures.py                    # Publication figures from multirun data
├── paper/
│   ├── main.tex                           # Paper source
│   └── references.bib                     # Bibliography
├── figures_multirun/                      # Generated figures (after running)
├── results_multirun/                      # Aggregate JSON + report (after running)
├── requirements.txt
└── README.md
```

## Requirements

- Python ≥ 3.10
- CPU only, 16 GB RAM
- ~2 GB disk for GPT-2 model download (first run)

```bash
pip install -r requirements.txt
```

**Dependencies:** PyTorch, Transformers (HuggingFace), scikit-learn, NumPy, SciPy, matplotlib, wordfreq

## Running

### Multi-run stability analysis (recommended)

The multi-run wrapper is the primary entry point. It runs both experiments K times with different generation seeds, then aggregates results.

```bash
# Edit K at line 43 of run_multirun.py (default: 10, paper uses 20)
python run_multirun.py
# Output: ./results_multirun/
```

Runtime estimate: ~10 min per run × K runs (K=10 → ~1.5h, K=20 → ~3h on 4-core CPU).

### Individual experiments (single run)

The pipeline scripts can also be run standalone for a single generation:

```bash
# Experiment 1: Static embeddings
python hallucination_induction.py
# Output: ./results_induction/

# Experiment 2: Contextual hidden states
python hallucination_induction_contextual.py
# Output: ./results_induction_contextual/
```

### Figures

After running the multi-run analysis:

```bash
python generate_figures.py
# Output: ./figures_multirun/
```

Produces four publication figures:
- `fig1_forest.pdf` — Effect-size stability forest plot (main text)
- `fig2_pseudorep.pdf` — Pseudoreplication inflation dumbbell chart (main text)
- `fig3_ctx_scatter.pdf` — Contextual norm–H scatter from representative run (appendix)
- `fig4_pvalue_strips.pdf` — p-value distributions across K runs (appendix)

## Experimental Design

**Model:** GPT-2-small (124M parameters, `gpt2`), CPU only, temperature 1.0, no top-k/top-p.

**Prompts:** 15 per condition × 60 tokens each → ~900 tokens/condition, ~2,700 total.

**Two-level inference:** Prompt-level means (N=15/group) are the unit of analysis, not individual tokens. This eliminates pseudoreplication from within-prompt autocorrelation.

**Multi-run protocol:** Calibration is performed once (seed 42). Generation is repeated K times with seeds 1…K. Statistical analysis (permutation tests, bootstrap CIs) uses a fixed internal seed for determinism.

## Key Outputs

**`run_multirun.py`** produces:
- `multirun_aggregate.json` — full per-run statistics + cross-run aggregates
- `multirun_report.txt` — human-readable stability report
- `raw_results_contextual.json` — per-token data from representative run (for figures)
- `summary_contextual.json` — zone thresholds from representative run

**`hallucination_induction.py`** produces:
- `induction_report.txt` — full results with confusion matrix
- `raw_results.json` — per-token geometric measurements

**`hallucination_induction_contextual.py`** produces:
- `induction_contextual_report.txt` — full results with confusion matrix
- `raw_results_contextual.json` — per-token data with hidden-state metrics
- `summary.json` — zone thresholds and calibration stats

## Reproducibility

Calibration and statistical analysis use `random_state=42`. Text generation is stochastic by design (temperature 1.0); the multi-run protocol quantifies this variance rather than hiding it behind a fixed seed. No GPU required. All experiments run on a single Intel i7-6700 (4 cores, 16 GB RAM).

## License

Apache 2.0
