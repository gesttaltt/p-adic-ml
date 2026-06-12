# Conditional Metric Alignment — Hierarchical VQ-VAE

Checkpoint: `./checkpoints/hierarchical_bucket_attn/vqvae.pt`  N=64  300 samples/type/prime

## 1. Top-Code Branch Alignment

| Prime | Within-bucket dist | Cross-bucket dist | Ratio (within/cross) |
| :---: | :---: | :---: | :---: |
| $p=2$ | 0.6675 | 0.7032 | 0.949 (✓ tight) |
| $p=3$ | 0.7940 | 0.7553 | 1.051 (✗ loose) |
| $p=5$ | 0.8364 | 0.8203 | 1.020 (✗ loose) |
| $p=7$ | 0.8799 | 0.8702 | 1.011 (✗ loose) |
| $p=11$ | 0.9134 | 0.9209 | 0.992 (✓ tight) |

## 2. Conditional Bottom-Code Alignment

| Prime | Unconditional $r$ | Conditional $r$ (mean) | Gain | Buckets used |
| :---: | :---: | :---: | :---: | :---: |
| $p=2$ | 0.2939 | 0.2725 | -0.0214 ↓ | 4 |
| $p=3$ | 0.1579 | 0.1772 | +0.0193 ≈ | 8 |
| $p=5$ | 0.0800 | 0.0739 | -0.0061 ≈ | 13 |
| $p=7$ | 0.0528 | 0.0438 | -0.0090 ≈ | 15 |
| $p=11$ | 0.0347 | 0.0204 | -0.0142 ≈ | 15 |
