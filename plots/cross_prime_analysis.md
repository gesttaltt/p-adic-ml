# Cross-Prime Interpolation Analysis

Models: ['Euclidean hd=64', 'Poincaré hd=256']  
Prime pairs: [(2, 5), (2, 11), (5, 11)]  
Pairs averaged: 60, Steps: 11


## Euclidean hd=64

| Prime pair | Entropy t=0 | Entropy t=0.5 | Entropy t=1 | Dist-start t=0.5 | Dist-end t=0.5 | Monotone? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| $p=2 \to p=5$ | 1.291 | 1.395 | 1.365 | 0.5525 | 0.7970 | Yes |
| $p=2 \to p=11$ | 1.577 | 1.653 | 1.672 | 0.7573 | 0.8850 | Yes |
| $p=5 \to p=11$ | 1.513 | 1.582 | 1.672 | 0.7539 | 0.8187 | Yes |

## Poincaré hd=256

| Prime pair | Entropy t=0 | Entropy t=0.5 | Entropy t=1 | Dist-start t=0.5 | Dist-end t=0.5 | Monotone? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| $p=2 \to p=5$ | 1.299 | 1.476 | 1.444 | 0.5190 | 0.6847 | Yes |
| $p=2 \to p=11$ | 1.843 | 2.062 | 2.118 | 0.7298 | 0.7562 | Yes |
| $p=5 \to p=11$ | 2.037 | 2.102 | 2.118 | 0.7655 | 0.7021 | Partial |
