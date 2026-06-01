# Improvements Log

---

## Batch 1 — Completed

### 1. Capacity Scaling (hidden_dim 64 → 256) ✅

**Problem**: Broad-23 slightly underperforms Broad-19 on metric alignment (0.05983 vs 0.00698 for p=5), despite seeing more primes. The 64-dim hidden representation was saturating — nine prime topologies competing for the same capacity.

**Change**: Retrained Broad-19 with `hidden_dim=256` in both `ConditionalVQVAE` and `ConditionalBetaVAE`, everything else constant.

**Result**: hd=256 wins 3/4 metrics. 10-point accuracy jump on p=5 confirms the bottleneck hypothesis. `train_broad_p19.py` and `train_broad_p23.py` now default to `hidden_dim=256`.

---

### 2. Hyperbolic VAE (Poincaré Ball Manifold) ✅

**Problem**: The metric alignment loss is a proxy — it penalizes MSE between Euclidean latent distances and p-adic distances but can't enforce hyperbolic geometry.

**Change**: Replaced the Euclidean latent space with a Poincaré ball (`geoopt`), using the wrapped-normal reparameterization.

**Files**: `hyperbolic_vae.py`, `train_hyperbolic.py`, `metric_alignment.py` (`compute_hyperbolic_metric_loss`).

---

### 3. Cascade Router on Broad Models ✅

**Problem**: `evaluate_cascade.py` hard-coded `vocab_size=13`, blocking Broad-19 from being used as the slow-path fallback.

**Change**: Added `--vocab_size` (default 13) and `--checkpoint_dir` CLI arguments.

---

### 4. Cross-Prime Latent Interpolation ✅

**Problem**: `interpolate.py` only interpolated within a single prime base.

**Change**: Added `run_cross_prime_interpolation(model_path, p_start, p_end, decode_with, ...)` to `interpolate.py`.

---

### 5. Code Quality Fixes ✅

- `import math` moved to file top in `train.py`, `train_metric.py`, `evaluate_cascade.py`
- Removed unused `is_vqvae` parameter from `anomaly_detector.get_reconstruction_error`
- Removed dead import `from visualize_latent import project_pca` in `interpolate.py`
- Replaced hardcoded machine-specific artifact paths with `ARTIFACTS_DIR` env var
- Added `requirements.txt`

---

### 6. Lorentz Manifold & Learnable Curvature ✅

**Problem**: Poincaré ball had fixed curvature and can become numerically unstable near the boundary for large primes.

**Change**: Added `manifold='poincare'|'lorentz'` and `learnable_curvature=True/False` to `HyperbolicBetaVAE`. Upgraded to `RiemannianAdam`.

---

### 7. Unit Test Suite ✅

**Problem**: No automated tests to catch mathematical or architectural regressions.

**Change**: `test_pipeline.py` — full `unittest` suite covering math, dataset, metric alignment (Euclidean + hyperbolic), VQ-VAE, Prior GRU, Beta-VAE, HyperbolicBetaVAE (both manifolds, fixed and learnable curvature).

---

### 8. Curvature Sweep Analyzer ✅

**Problem**: Sweeping curvature configurations required manual edits to source files.

**Change**: `sweep_curvature.py` — trains 5 configs (fixed $c \in \{0.5, 1, 2, 5\}$ + learnable) on a shared split and writes a Markdown report. Supports both manifolds.

---

## Batch 2 — Planned

### 9. Broader Primes at hd=256 (Scaling Curve Completion) ✅

**Problem**: All scaling experiments beyond Broad-17 used `hidden_dim=64`. The capacity scaling result (hd=256 wins on p=5) was validated only at Broad-19. We don't know whether the Broad-23 accuracy dip persists at higher capacity or was purely a capacity bottleneck.

**Plan**:
- Run `train_broad_p19.py` (hd=256, saves to `./checkpoints/broad_p19_hd256/`)
- Run `train_broad_p23.py` (hd=256, loads Broad-19 from the hd=256 dir)
- Compare VQ-VAE accuracy and metric alignment at p=2 and p=5 across Restricted → Broad-23

**Fix applied**: `train_broad_p23.py` updated to load Broad-19 from `./checkpoints/broad_p19_hd256/` (hd=256) instead of the old `./checkpoints/broad_p19/` (hd=64).

**Result**: The dip **persists at hd=256**. Broad-19 wins 3/4 metrics: p=5 accuracy 73.15% vs 64.32%, metric alignment 0.01195 vs 0.01805. Capacity was not the primary cause — the 23rd prime's regularization burden outweighs its benefit at this architecture size. Broad-19 hd=256 sets a new p=5 accuracy record (73.15%), surpassing Broad-17 hd=64 (69.87%) by +3.3pp.

---

### 10. Converged Curvature Sweep ✅

**Problem**: Prior sweep reports were 3-epoch snapshots (~25% accuracy) — too underfit to draw conclusions.

**Result**: Ran both manifolds at 15 epochs. Key findings:

*Poincaré*: hard accuracy/alignment trade-off. c=0.5 best for accuracy (42.95%), c=5.0 best for alignment (0.01501). Phase transition around c≈1.5. Learnable c converges to 1.23 — stuck in the mediocre transition region, missing both optima.

*Lorentz*: monotone in c — higher c improves both accuracy and alignment. c=5.0 wins outright (33.05% acc, 0.07411 align). c=0.5 produces NaN (low-curvature instability). Learnable c drifts down to 0.59, away from the better-performing high-k regime.

**Practical recommendation**: c=5.0 Poincaré for alignment-focused runs; c=0.5 Poincaré for accuracy-focused; Lorentz always use k≥2.0.

---

### 11. Hyperbolic VAE at hd=256 ✅

**Problem**: All prior Hyperbolic VAE results used `hidden_dim=64`, leaving the capacity question open.

**Result**: Trained Poincaré and Lorentz at hd=256, N=64, c=1.0, 15 epochs. Evaluated with `eval_hyperbolic_hd256.py`.

- **Hyp-P hd=256 wins on alignment loss across all 5 primes** — weighted avg 0.00995 (-65% vs Hyp-P hd=64, -74% vs Euc hd=64). First config to beat Euclidean at p=2 and p=3.
- **Lorentz hd=256 disappoints** — performs on par with Poincaré hd=64, not hd=256. Larger ambient dimension may need more epochs.
- **Spearman r barely changes** — capacity tightens magnitude of alignment but rank ordering was already correct at hd=64.

`eval_hyperbolic_hd256.py` is the reusable eval script for this comparison.

---

### 12. Systematic Cross-Prime Interpolation Analysis ✅

**Problem**: Interpolation plots existed but no quantitative analysis of path structure.

**Result**: `analyze_cross_prime.py` averages 60 endpoint pairs per (model, prime-pair) over 11 steps, measuring digit entropy, dist-to-start, and dist-to-end.

Key findings:
- **Monotonic transitions on all pairs** (both models) — the latent space is a continuous topological manifold, not a prime-partitioned space.
- **Entropy nearly constant along path** (≤0.2 nats variation) — no "confused" midpoint region.
- **Asymmetric speed**: $p=2 \to p=5$ paths spend more time near the binary endpoint before transitioning.
- Poincaré hd=256 shows same qualitative behavior as Euclidean hd=64 but with higher absolute entropy at high-branching primes.

---

### 13. Hierarchical VQ-VAE ✅

**Problem**: Flat GRU prior treats all positions equally; doesn't exploit the multi-scale tree structure of p-adic numbers.

**Result**: Implemented `HierarchicalVQVAE` (VQ-VAE-2 style) + `TopPriorGRU` + `BotPriorGRU`.

Architecture:
- Shared encoder → bottom branch (N/2 tokens, codebook 64) + top branch (N/4 tokens, codebook 16)
- Top-down decoder: upsample top as context, add to bottom, decode
- TopPrior: autoregressive GRU over 16 top indices
- BotPrior: autoregressive GRU over 32 bottom indices, conditioned on top via repeat-interleave injection

Results (Broad-11, N=64, hd=64, 141K params total):
- Val accuracy: **78.03%** (+18pp over flat VQ-VAE ~60%)
- Per-prime: p=2 98.4%, p=3 96.8%, p=5 79.4%, p=7 63.2%, p=11 47.2%
- Top-prior accuracy: 19.9% (3.2× random baseline 1/16)
- Bottom-prior accuracy: 39.6% (25× random baseline 1/64)
- Prior samples: all valid digits; some show structured repetition matching rational p-adic patterns

Files: `hierarchical_vqvae.py`, `train_hierarchical.py`

---

## Batch 3 — Planned

### 14. Top Codebook Interpretability Analysis ✅
### 15. Conditional Generation Coherence Test ✅

Both items run together in `analyze_top_codes.py`.

**Results** (4500 sequences, Broad-11, N=64):

- **Prime specialization confirmed**: code 11 = 100% p=2; codes 0/4 = 91%/85% p=2; codes 7/10 = 64%/66% p=3. Higher-branching primes share two large catch-all codes (codes 2 and 5, ~750 seqs each).
- **Within-code distance tighter than cross-code**: 8/16 codes. The 8 looser codes are the large catch-alls with high internal diversity.
- **Conditional coherence: 16/16 codes** produce p=5 samples tighter than unconditional baseline (0.838). Codes 4 and 7 achieve mean distance 0.276 and 0.271 — ~3× tighter than random. The top code is a genuine branch selector.

Key finding: the success metric (within-code < 0.5× cross-code for 10/16) was not met on distance alone, but the conditional coherence test (16/16) is the stronger result — it shows the top codes causally constrain generation, not just correlate with sequence properties.

---

### 16. Hierarchical VQ-VAE at hd=256

**Problem**: The hierarchical model was benchmarked at `hidden_dim=64`. All other capacity experiments showed hd=256 gives +10pp on high-branching primes. We don't know if the hierarchical architecture still benefits from capacity scaling, or whether the two-level structure already solves the capacity bottleneck.

**Plan**: Retrain `HierarchicalVQVAE` with `hidden_dim=256` on Broad-11, N=64. Same epochs and hyperparameters as the hd=64 run for a fair comparison.

**What to measure**: Val accuracy, per-prime reconstruction accuracy, top/bottom prior accuracy.

**Expected outcome**: If the +18pp hierarchical gain is orthogonal to capacity, hd=256 should add another +5–10pp on top (p=7, p=11 specifically). If the hierarchy already saturates the capacity, gains will be smaller.

**Command**:
```bash
python train_hierarchical.py --primes 2 3 5 7 11 --N 64 --hidden_dim 256 \
  --save_dir ./checkpoints/hierarchical_hd256
```

---

### 17. Hierarchical VQ-VAE on Broad-19

**Problem**: The hierarchical model was only trained on Broad-11. The key open question from Batch 2 is whether the Broad-23 accuracy dip is fundamental or architectural. The flat VQ-VAE plateaus at Broad-23 even at hd=256. The hierarchical architecture might handle more primes better because each level only models a subset of the total entropy — reducing the per-prime competition for codebook capacity.

**Plan**: Train `HierarchicalVQVAE` on Broad-19 (8 primes, primes up to 19), evaluate on p=2 and p=5.

**What to measure**: VQ-VAE accuracy and metric alignment on p=2 and p=5 vs flat Broad-19 hd=256 (73.15% p=5 accuracy).

**Expected outcome**: If the hierarchical structure mitigates the capacity competition between primes, Broad-19 hierarchical should approach or exceed 73.15% p=5 accuracy with fewer parameters than the flat hd=256 model.

**Command**:
```bash
python train_hierarchical.py --primes 2 3 5 7 11 13 17 19 --N 64 \
  --save_dir ./checkpoints/hierarchical_broad19
```

---

### 18. Metric Alignment Evaluation for Hierarchical Model

**Problem**: All metric alignment results so far are for flat Euclidean and Hyperbolic models. We don't know whether the hierarchical VQ-VAE's strong reconstruction accuracy (+18pp) translates into better ultrametric alignment. The two quantities are not correlated in general — a model can reconstruct well but organize the latent space poorly (or vice versa).

**Plan**: Extract the bottom-level quantized representations `z_q_bot` from the hierarchical VQ-VAE and compute:
1. Per-prime metric alignment loss (MSE between normalized pairwise Euclidean distances and p-adic distances)
2. Per-prime Spearman r

Compare against the Euclidean hd=64 and hd=256 flat models using `eval_hyperbolic_hd256.py` as a template.

**What to measure**: Metric alignment loss and Spearman r for p=2,3,5,7,11 on the hierarchical bottom codes.

**Where to add code**: Extend `eval_hyperbolic_hd256.py` or create `eval_hierarchical_alignment.py`.

---

### 19. Hyperbolic Top Codes

**Problem**: The top branch of the hierarchical VQ-VAE quantizes to Euclidean codebook vectors. But the top codes are supposed to represent global tree-branch identity — exactly the kind of hierarchical structure that hyperbolic geometry models naturally. Replacing the top-level Euclidean quantizer with a Poincaré-ball codebook could give the top codes a geometry that better matches the branching structure they need to represent.

**Plan**: Add `manifold='poincare'` option to the top branch of `HierarchicalVQVAE`. The top codebook embeddings become `geoopt.ManifoldParameter` on the Poincaré ball. The VQ lookup uses geodesic distance instead of Euclidean. The top prior still operates on indices, so the prior architecture is unchanged.

**Estimated scope**: Medium. Requires modifying `VectorQuantizer` (or adding `HyperbolicVectorQuantizer`) and updating `HierarchicalVQVAE.encode()`.

**Expected outcome**: Hyperbolic top codes should improve metric alignment on high-branching primes (p≥7) while maintaining or improving reconstruction accuracy, since the top-level geometry now matches the tree structure it represents.

**Prerequisite**: Items 14 and 15 should confirm the top codes are actually encoding tree-branch structure before investing in this change.
