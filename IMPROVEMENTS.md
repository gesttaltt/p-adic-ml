# Planned Improvements

Tracked improvements to implement after the continuous prime embedding refactor.

---

## 1. Capacity Scaling (hidden_dim 64 → 256) ✅ wired up

**Problem**: The Broad-23 model slightly underperforms Broad-19 on metric alignment loss (0.05983 vs 0.00698 for p=5), despite seeing more training primes. This looks like the shared 64-dim hidden representation saturating — nine prime topologies competing for the same capacity.

**Proposed change**: Retrain Broad-19 (the best configuration) with `hidden_dim=256` in both `ConditionalVQVAE` and `ConditionalBetaVAE`. Everything else stays constant.

**What to measure**: VQ-VAE accuracy on p=2 and p=5, metric alignment loss on both, vs baseline Broad-19.

**Status**: `train_broad_p19.py` updated — `hidden_dim=256` is now the default in the config block, and the save directory is `./checkpoints/broad_p19_hd256`. Run `python scaling_analysis/train_broad_p19.py` to execute.

---

## 2. True Hyperbolic VAE (Poincaré Ball Manifold) ✅ done

**Problem**: The metric alignment loss in `metric_alignment.py` is a proxy: it penalizes MSE between Euclidean latent distances and p-adic distances. The latent space is still Euclidean (R^32) — the loss pulls it toward hyperbolic structure but can't enforce it geometrically.

**Proposed change**: Replace the Euclidean latent space in `ConditionalBetaVAE` with a Poincaré ball (using the `geoopt` library for PyTorch Riemannian manifolds). The reparameterization trick becomes a wrapped-normal distribution on the manifold. Drop the metric alignment loss entirely — the geometry enforces the ultrametric automatically.

**References**:
- geoopt: `pip install geoopt`
- Wrapped Normal on Poincaré Ball: Mathieu et al. 2019 "Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders"

**Status**: Done.
- `hyperbolic_vae.py` — `HyperbolicBetaVAE`: Poincaré-ball latent space, wrapped-normal reparameterize, logmap0 decoder
- `train_hyperbolic.py` — standalone training script (loss = recon + β‖μ‖² + γ·hyperbolic_metric_alignment)
- `metric_alignment.py` — added `compute_hyperbolic_metric_loss` using geodesic distance
- Run: `python train_hyperbolic.py [--primes ...] [--curvature 1.0] [--gamma 5.0]`

---

## 3. Cascade Router Evaluation on Broad Models ✅ done

**Problem**: `evaluate_cascade.py` instantiates models with `vocab_size=13`, hardcoding it to the Broad-11 family. The best model (Broad-19, vocab_size=19) cannot be used as the slow-path fallback. We don't know if better metric alignment in Broad-19 actually improves cascade routing quality.

**Proposed change**: Add `--vocab_size` and `--checkpoint_dir` arguments to `evaluate_cascade.py`. Update all `ConditionalVQVAE`, `ConditionalBetaVAE`, and `PriorGRU` instantiations in that file to use the argument.

**Status**: Done. `evaluate_cascade.py` now accepts `--vocab_size` (default 13) and `--checkpoint_dir`. Example usage for Broad-19:
```bash
python evaluate_cascade.py \
  --primes 2 3 5 7 11 13 17 19 \
  --vocab_size 19 \
  --checkpoint_dir ./checkpoints/broad_p19
```

---

## 4. Cross-Prime Latent Interpolation ✅ done

**Problem**: `interpolate.py` only interpolates within a single prime base. The Broad-19 model shares a single latent space across all primes — it should be possible to interpolate between a p=2 and a p=5 sequence. The intermediate decoded sequences could reveal whether the model learned cross-base topology.

**Proposed change**: Extend `interpolate.py` with a `run_cross_prime_interpolation(p_start, p_end, ...)` function. Encode one sequence from each base, interpolate z_1 → z_2, and decode each intermediate z with a target prime (e.g. always decode with p=5). Log what digit structure appears mid-path.

**Status**: Done. `run_cross_prime_interpolation(model_path, p_start, p_end, decode_with, ...)` added to `interpolate.py`. The `__main__` block auto-runs it using `./checkpoints/embedding_comparison/continuous/beta_vae_metric.pt` if present.

---

## 5. Code Quality Fixes ✅ done

These are small, independent cleanups with no research impact.

### 5a. `import math` inside training loops
`train.py:42`, `train_metric.py:43`, `evaluate_cascade.py:44` all have `import math` inside the per-batch loop. Move to file top.

### 5b. Dead code removal
- `anomaly_detector.get_reconstruction_error`: `is_vqvae` parameter is accepted but never read — the function calls `model(digits, p)` regardless. Remove the parameter.
- `interpolate.py:8`: `from visualize_latent import project_pca` is imported but never called (the file defines its own PCA helpers). Remove the import.

### 5c. Hardcoded artifact copy paths
`scaling_analysis/train_broad_p23.py:357-361` and `scaling_analysis/visualize_scaling_trees.py:89-92` and `scaling_analysis/poincare_embedding.py:185-188` all do `os.system(f"cp ... /home/gestalt/.gemini/antigravity-cli/brain/...")`. This path is machine-specific and will silently fail on any other machine. Remove these lines entirely, or replace with an optional `--artifacts_dir` CLI argument.

### 5d. Add `requirements.txt` ✅
Added `requirements.txt` with `torch>=2.0`, `matplotlib>=3.7`, `numpy>=1.24`.
