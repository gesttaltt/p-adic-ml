# Improvements Log

All tracked improvement items are complete. This file records what was done and why. Open research directions are maintained in the README.

---

## 1. Capacity Scaling (hidden_dim 64 → 256) ✅

**Problem**: The Broad-23 model slightly underperforms Broad-19 on metric alignment loss (0.05983 vs 0.00698 for p=5), despite seeing more training primes. This looks like the shared 64-dim hidden representation saturating — nine prime topologies competing for the same capacity.

**Change**: Retrained Broad-19 with `hidden_dim=256` in both `ConditionalVQVAE` and `ConditionalBetaVAE`, everything else constant.

**Result**: hd=256 wins 3/4 metrics. 10-point accuracy jump on p=5 confirms the capacity bottleneck hypothesis. `train_broad_p19.py` and `train_broad_p23.py` now default to `hidden_dim=256`, saving to `./checkpoints/broad_p19_hd256` and `./checkpoints/broad_p23_hd256`.

---

## 2. Hyperbolic VAE (Poincaré Ball Manifold) ✅

**Problem**: The metric alignment loss in `metric_alignment.py` is a proxy: it penalizes MSE between Euclidean latent distances and p-adic distances. The latent space is still Euclidean — the loss pulls it toward hyperbolic structure but can't enforce it geometrically.

**Change**: Replaced the Euclidean latent space in `ConditionalBetaVAE` with a Poincaré ball (`geoopt`). The reparameterization trick is the wrapped-normal distribution on the manifold.

**Files**:
- `hyperbolic_vae.py` — `HyperbolicBetaVAE`: Poincaré-ball latent space, wrapped-normal reparameterize, logmap0 decoder
- `train_hyperbolic.py` — standalone training script (loss = recon + β‖μ‖² + γ·hyperbolic_metric_alignment)
- `metric_alignment.py` — added `compute_hyperbolic_metric_loss` using geodesic distance

---

## 3. Cascade Router Evaluation on Broad Models ✅

**Problem**: `evaluate_cascade.py` instantiated models with `vocab_size=13`, hardcoding it to the Broad-11 family. The best model (Broad-19) could not be used as the slow-path fallback.

**Change**: Added `--vocab_size` (default 13) and `--checkpoint_dir` arguments. All model instantiations use the argument.

```bash
python evaluate_cascade.py \
  --primes 2 3 5 7 11 13 17 19 \
  --vocab_size 19 \
  --checkpoint_dir ./checkpoints/broad_p19
```

---

## 4. Cross-Prime Latent Interpolation ✅

**Problem**: `interpolate.py` only interpolated within a single prime base. The Broad-19 model shares one latent space across all primes, so cross-base paths should be possible.

**Change**: Added `run_cross_prime_interpolation(model_path, p_start, p_end, decode_with, ...)` to `interpolate.py`. The `__main__` block auto-runs it using `./checkpoints/embedding_comparison/continuous/beta_vae_metric.pt` if present.

---

## 5. Code Quality Fixes ✅

- `import math` moved from inside training loops to file top in `train.py`, `train_metric.py`, `evaluate_cascade.py`
- `anomaly_detector.get_reconstruction_error`: removed unused `is_vqvae` parameter
- `interpolate.py`: removed unused `from visualize_latent import project_pca` import
- Hardcoded machine-specific artifact copy paths removed from `train_broad_p19.py`, `train_broad_p23.py`, `visualize_scaling_trees.py`, `poincare_embedding.py`; scripts now use `ARTIFACTS_DIR` env var
- `requirements.txt` added with `torch>=2.0`, `matplotlib>=3.7`, `numpy>=1.24`

---

## 6. Lorentz Manifold & Learnable Curvature ✅

**Problem**: The Poincaré ball had fixed curvature $c=1.0$ and can become unstable near the boundary for denser branching factors.

**Change**: Added `manifold='poincare'|'lorentz'` and `learnable_curvature=True/False` to `HyperbolicBetaVAE`. Unified manifold math functions (`origin`, `proju`, `transp0`, `expmap`, `projx`) to work across both. Optimizer upgraded to `geoopt.optim.RiemannianAdam` so the curvature parameter is optimized on its manifold.

```bash
python train_hyperbolic.py --manifold lorentz --learnable_curvature
```

---

## 7. Unit Test Suite ✅

**Problem**: No automated test suite existed to catch mathematical or architectural regressions.

**Change**: `test_pipeline.py` — `unittest`-based suite covering modular arithmetic (`mod_inverse`), p-adic conversions, Hensel lifting, dataset generation, Euclidean/hyperbolic metric alignment, VQ-VAE, Prior GRU, Euclidean Beta-VAE, and Hyperbolic VAE (Poincaré and Lorentz, fixed and learnable curvature).

```bash
python test_pipeline.py
```

---

## 8. Curvature Sweep Analyzer ✅

**Problem**: Sweeping curvature configurations was manual and required editing source files.

**Change**: `sweep_curvature.py` — trains 5 configurations (fixed $c \in \{0.5, 1.0, 2.0, 5.0\}$ and learnable $c=1.0$) on a shared dataset split and saves a Markdown report. Works for both `--manifold poincare` and `--manifold lorentz`. Reports saved to `scaling_analysis/curvature_sweep_{poincare,lorentz}.md`.

```bash
python sweep_curvature.py --manifold poincare --epochs 15
python sweep_curvature.py --manifold lorentz --epochs 15
```
