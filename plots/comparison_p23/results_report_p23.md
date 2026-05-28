# Six-Way Comparison: Scaling Analysis up to p=23

This report evaluates the scaling effects of multi-task regularization in conditional p-adic models across six configurations.

## Evaluation Metrics Summary Table

| Evaluation Metric | Restricted | Broad-11 | Broad-13 | Broad-17 | Broad-19 | Broad-23 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `vae_metric_p2` | 0.02469 | 0.01207 | 0.00947 | 0.01073 | 0.00827 | 0.01262 |
| `vae_metric_p5` | 0.06533 | 0.01887 | 0.02559 | 0.02520 | 0.00698 | 0.05983 |
| `vq_acc_p2` | 0.98581 | 0.99039 | 0.97388 | 0.99333 | 0.98526 | 0.98010 |
| `vq_acc_p5` | 0.51701 | 0.59984 | 0.64518 | 0.69872 | 0.67529 | 0.68318 |
