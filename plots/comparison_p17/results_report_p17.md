# Four-Way Comparison: Restricted [2, 5], Broad-11, Broad-13, and Extended Broad-17 Models

This report evaluates the scaling effects of multi-task regularization in conditional p-adic models. We compare a restricted model (2 primes), two broad models (5 and 6 primes), and a newly extended broad model (7 primes, adding $p=17$).

## Evaluation Metrics Summary Table

| Evaluation Metric | Restricted Model [2, 5] | Broad-11 Model [2..11] | Broad-13 Model [2..13] | Broad-17 Model [2..17] |
| :--- | :---: | :---: | :---: | :---: |
| `prior_periodic_p2` | 0.87500 | 0.83500 | 0.85000 | 0.89500 |
| `prior_periodic_p5` | 0.22000 | 0.23000 | 0.27000 | 0.33000 |
| `prior_unique_p2` | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| `prior_unique_p5` | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| `prior_valid_p2` | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| `prior_valid_p5` | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| `vae_metric_p2` | 0.02524 | 0.01222 | 0.00958 | 0.01077 |
| `vae_metric_p5` | 0.06432 | 0.01887 | 0.02527 | 0.02504 |
| `vae_recon_p2` | 0.37069 | 0.37235 | 0.43393 | 0.52314 |
| `vae_recon_p5` | 1.19884 | 1.04776 | 1.16281 | 1.21442 |
| `vq_acc_p2` | 0.98510 | 0.98956 | 0.97185 | 0.99221 |
| `vq_acc_p5` | 0.51878 | 0.60219 | 0.64729 | 0.69969 |
