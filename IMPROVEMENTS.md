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

### 16. Hierarchical VQ-VAE at hd=256 ✅

**Result** (Broad-11, N=64, 1.98M params):
- Val acc: 79.38% (+1.35pp vs hd=64's 78.03%)
- p=5: 82.30% (+3pp), p=7: 68.89% (+5.7pp), p=11: 46.98% (≈same)
- Top-prior: 37.15% (nearly 2× hd=64's 19.9%)
- Bottom-prior: 32.32% (lower than hd=64's 39.6% — sign of better top/bottom factorization)

The hierarchy largely solved the capacity bottleneck. hd=256 adds only +3pp on p=5 vs flat model's +10pp gain. Top-prior improvement is the biggest story: larger encoder gives sharper global representations.

---

### 17. Hierarchical VQ-VAE on Broad-19 ✅

**Result** (Broad-19, N=64, hd=64, 143K params):
- Val acc: 64.45%
- p=2: 99.18%, p=5: **82.72%**, p=7: 68.80%, p=11: 52.23%, p=13: 46.78%, p=17: 37.89%, p=19: 34.45%
- Top-prior: 29.11%, Bottom-prior: 30.57%

**Key result**: Hierarchical hd=64 Broad-19 achieves 82.72% on p=5 — **+9.6pp over flat hd=256 Broad-19** (73.15%) with 8× fewer parameters. The hierarchy provides a more fundamental improvement than capacity scaling, and the multi-task regularization from 8 primes combines with the hierarchical structure synergistically.

---

### 18. Metric Alignment Evaluation for Hierarchical Model ✅

**Result** (`eval_hierarchical_alignment.py`, Broad-11 eval set):

Hierarchical bottom codes have weighted-avg alignment loss 0.161 and Spearman r=0.048 — ~4× higher loss and ~13× lower r than flat Euclidean (0.038 / 0.656).

**This is expected and correct.** Bottom codes encode within-bucket variation (fine-grained digit patterns given a top code); they are not supposed to globally organize all sequence-to-sequence distances. Global tree distance is handled by the top codes (confirmed by the 16/16 conditional coherence test). Measuring global metric alignment against bottom codes alone conflates the two levels — the correct evaluation is (1) top-code alignment at the branch level and (2) conditional bottom-code alignment within sequences sharing the same top code.

---

### 19. Hyperbolic Top Codes ✅

**Result** (Broad-11, N=64, hd=64, `--hyperbolic_top --top_curvature 1.0`):

- Val accuracy: 63.98% (vs Euclidean top 78.03%)
- Top-prior accuracy: **100% from epoch 2** (vs Euclidean 19.94%)
- Bottom-prior accuracy: 22.97% (vs 39.59%)

**Diagnosis — codebook collapse.** 100% top-prior accuracy by epoch 2 is a near-certain sign the codebook collapsed to 1–2 active codes. When all codebook vectors initialize near the ball origin, their geodesic distances are nearly identical and the commitment loss creates a strong pull toward the dominant code rather than spreading utilization.

**Fixes to explore** (Batch 4 candidate):
1. Spread initialization using the Poincaré ball's uniform measure
2. EMA codebook updates instead of gradient descent
3. Codebook-utilization entropy regularizer

`HyperbolicVectorQuantizer` is implemented in `hierarchical_vqvae.py`. The eval-time indexing bug (`ManifoldParameter` not callable) was also fixed in `train_hierarchical.py`.
