# Conditional Metric Alignment — Hierarchical VQ-VAE

Checkpoint: `./checkpoints/hierarchical/vqvae.pt`  N=64  300 samples/type/prime

## 1. Top-Code Branch Alignment

| Prime | Within-bucket dist | Cross-bucket dist | Ratio (within/cross) |
| :---: | :---: | :---: | :---: |
| $p=2$ | 0.5770 | 0.6647 | 0.868 (✓ tight) |
| $p=3$ | 0.7383 | 0.7744 | 0.953 (✓ tight) |
| $p=5$ | 0.8369 | 0.8652 | 0.967 (✓ tight) |
| $p=7$ | 0.8412 | 0.8635 | 0.974 (✓ tight) |
| $p=11$ | 0.9043 | 0.9282 | 0.974 (✓ tight) |

## 2. Conditional Bottom-Code Alignment

| Prime | Unconditional $r$ | Conditional $r$ (mean) | Gain | Buckets used |
| :---: | :---: | :---: | :---: | :---: |
| $p=2$ | 0.0900 | 0.1270 | +0.0370 ↑ | 8 |
| $p=3$ | 0.0630 | 0.0578 | -0.0052 ≈ | 14 |
| $p=5$ | 0.0486 | 0.0269 | -0.0217 ↓ | 13 |
| $p=7$ | 0.0341 | 0.0714 | +0.0373 ↑ | 10 |
| $p=11$ | 0.0322 | 0.0339 | +0.0017 ≈ | 11 |
