# Cascade Router: Flat vs Hierarchical Slow Path

Fast path: Beta-VAE (`./checkpoints/euclidean_n64/beta_vae_metric.pt`)  
Precision metric: HierarchicalVQVAE recon accuracy  
N=64, 500 samples, primes=[2, 3, 5, 7, 11]

| τ | Flat fast% | Flat prec% | Hier fast% | Hier prec% |
| :---: | :---: | :---: | :---: | :---: |
| 0.00 | 0.0 | 77.53 | 0.0 | 93.94 |
| 0.10 | 0.0 | 77.92 | 0.0 | 93.92 |
| 0.25 | 0.0 | 77.73 | 0.0 | 94.07 |
| 0.40 | 3.4 | 77.80 | 4.8 | 94.05 |
| 0.60 | 20.0 | 77.91 | 20.0 | 93.89 |
| 0.80 | 39.8 | 77.26 | 39.6 | 93.25 |
| 1.00 | 55.2 | 77.15 | 55.2 | 91.22 |
| 1.50 | 81.4 | 76.37 | 81.8 | 84.12 |
| 2.00 | 100.0 | 75.62 | 100.0 | 76.17 |
| 3.00 | 100.0 | 75.62 | 100.0 | 76.17 |
| 10.00 | 100.0 | 75.62 | 100.0 | 76.17 |
