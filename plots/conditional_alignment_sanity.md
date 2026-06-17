# Conditional Metric Alignment — Hierarchical VQ-VAE

Checkpoint: `./checkpoints/hierarchical_bucket_attn_sanity/vqvae.pt`  N=64  300 samples/type/prime

## 1. Top-Code Branch Alignment

| Prime | Within-bucket dist | Cross-bucket dist | Ratio (within/cross) |
| :---: | :---: | :---: | :---: |
| $p=2$ | 0.6614 | 0.7000 | 0.945 (✓ tight) |
| $p=3$ | 0.7299 | 0.7711 | 0.947 (✓ tight) |
| $p=5$ | 0.8294 | 0.8220 | 1.009 (✗ loose) |
| $p=7$ | 0.8953 | 0.8494 | 1.054 (✗ loose) |
| $p=11$ | 0.9248 | 0.9249 | 1.000 (✓ tight) |

## 2. Conditional Bottom-Code Alignment

| Prime | Unconditional $r$ | Conditional $r$ (mean) | Gain | Buckets used |
| :---: | :---: | :---: | :---: | :---: |
| $p=2$ | 0.1241 | 0.1170 | -0.0071 ≈ | 5 |
| $p=3$ | 0.0502 | 0.1225 | +0.0723 ↑ | 3 |
| $p=5$ | 0.0327 | 0.0432 | +0.0104 ≈ | 5 |
| $p=7$ | 0.0247 | 0.0041 | -0.0206 ↓ | 5 |
| $p=11$ | 0.0185 | 0.0044 | -0.0141 ≈ | 5 |
