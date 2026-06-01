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

### 11. Hyperbolic VAE at hd=256

**Problem**: All Hyperbolic VAE results (both N=32 and N=64 tables in the README) used `hidden_dim=64`. The VQ-VAE benefits strongly from hd=256 (+10pp on p=5). The hyperbolic model faces the same capacity bottleneck, especially at high-branching primes where the latent space needs to represent more complex structure.

**Plan**: Retrain `HyperbolicBetaVAE` on Broad-11 at `hidden_dim=256, N=64`, both Poincaré and Lorentz. Compare metric alignment and Spearman r against the existing hd=64 Hyperbolic VAE and the hd=256 Euclidean Beta-VAE.

**What to measure**: Metric alignment loss and Spearman r per prime (p=2,3,5,7,11), both manifolds.

**Expected outcome**: hd=256 hyperbolic should win at p≥5 even more decisively, widening the gap over Euclidean hd=256.

**Command**:
```bash
python train_hyperbolic.py --primes 2 3 5 7 11 --N 64 --hidden_dim 256 --manifold poincare --save_dir ./checkpoints/hyperbolic_n64_hd256
python train_hyperbolic.py --primes 2 3 5 7 11 --N 64 --hidden_dim 256 --manifold lorentz  --save_dir ./checkpoints/lorentz_n64_hd256
```

---

### 12. Systematic Cross-Prime Interpolation Analysis

**Problem**: The infrastructure for cross-prime interpolation exists and the plots are generated, but there's no quantitative analysis of what the interpolated sequences look like. We don't know whether the latent path between p=2 and p=5 produces statistically meaningful in-between structure or just garbled noise.

**Plan**: For a grid of 11 interpolation steps (t=0.0 to 1.0):
1. Compute the digit-frequency histogram of decoded sequences at each step
2. Compute the pairwise p-adic distance distribution of the decoded sequences
3. Measure how many decoded sequences are valid (each digit < p for the target prime)
4. Plot how these statistics change along the path

**What to measure**: Digit-frequency entropy, mean p-adic distance to the t=0 endpoint, fraction of valid digits — all as a function of t.

**Expected outcome**: A monotonic transition in digit statistics would confirm the model learned a topologically meaningful shared representation. A sudden jump would indicate the latent space is prime-partitioned with a discontinuity.

**Where to add code**: Extend `interpolate.py` with a `analyze_cross_prime_path(...)` function that returns these statistics and saves a summary plot.

---

### 13. Hierarchical Prior

**Problem**: The current prior is a flat GRU over VQ-VAE tokens, treating all sequence positions equally. p-adic trees have explicit hierarchical structure — the first digit determines the top-level branch, and each subsequent digit refines within that branch. A flat prior can't directly model this multi-scale structure.

**Plan**: Implement a two-level VQ-VAE-2 style prior:
- **Top codebook** (size 16): captures the high-level branch (first 3–4 digits)
- **Bottom codebook** (size 64): captures local digit refinements, conditioned on the top code
- Each level gets its own small GRU prior

**What to measure**: VQ-VAE reconstruction accuracy, metric alignment, and — most importantly — the qualitative structure of prior-sampled sequences (do they look like coherent tree paths?).

**Estimated scope**: Large. Requires refactoring `models.py` (add `HierarchicalVQVAE`), `train.py` (two-pass training: bottom then top), and `anomaly_detector.py` (cascade router needs to handle hierarchical generation).

**Prerequisite**: Complete items 9–11 first. The hierarchical prior should be benchmarked against the best Euclidean and hyperbolic models at matched capacity.
