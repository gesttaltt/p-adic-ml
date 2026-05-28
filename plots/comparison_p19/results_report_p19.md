# Five-Way Comparison: Scaling Analysis up to p=19

This report evaluates the scaling effects of multi-task regularization in conditional p-adic models across five configurations.

## Evaluation Metrics Summary Table

| Evaluation Metric | Restricted | Broad-11 | Broad-13 | Broad-17 | Broad-19 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `vae_metric_p2` | 0.02494 | 0.01212 | 0.00940 | 0.01057 | 0.00815 |
| `vae_metric_p5` | 0.06537 | 0.01901 | 0.02639 | 0.02535 | 0.00731 |
| `vq_acc_p2` | 0.98583 | 0.98943 | 0.97268 | 0.99203 | 0.98617 |
| `vq_acc_p5` | 0.51826 | 0.60232 | 0.65151 | 0.70911 | 0.67953 |
