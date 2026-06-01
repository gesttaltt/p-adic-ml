# Top Codebook Interpretability Analysis

Model: `./checkpoints/hierarchical`  
Dataset: Broad-11, N=64, 300 samples/type/prime

Cross-code distance baseline: **0.7973**  
Unconditional coherence baseline (p=5): **0.8382**

| Code | N | Top prime | Top type | Common prefix | Within dist | vs baseline | Coh dist | Coh vs baseline |
| :---: | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 108 | p=2 (91%) | Random | [0, 0] | 0.6269 | ✓ tight | 0.8035 | ✓ tight |
| 1 | 103 | p=3 (37%) | Rational | [1, 1] | 0.7721 | ✓ tight | 0.7191 | ✓ tight |
| 2 | 753 | p=5 (36%) | Algebraic | [2, 0] | 0.8374 | ✗ loose | 0.4986 | ✓ tight |
| 3 | 160 | p=3 (36%) | Rational | [1, 1] | 0.8301 | ✗ loose | 0.6810 | ✓ tight |
| 4 | 171 | p=2 (85%) | Rational | [0, 0] | 0.6680 | ✓ tight | 0.2761 | ✓ tight |
| 5 | 752 | p=7 (36%) | Random | [0, 0] | 0.8210 | ✗ loose | 0.7136 | ✓ tight |
| 6 | 360 | p=2 (65%) | Algebraic | [0, 0] | 0.7966 | ✓ tight | 0.6220 | ✓ tight |
| 7 | 162 | p=3 (66%) | Rational | [2, 0] | 0.6953 | ✓ tight | 0.2711 | ✓ tight |
| 8 | 149 | p=2 (29%) | Random | [1, 1] | 0.7780 | ✓ tight | 0.6805 | ✓ tight |
| 9 | 93 | p=7 (40%) | Algebraic | [1, 1] | 0.7982 | ✗ loose | 0.4912 | ✓ tight |
| 10 | 133 | p=3 (64%) | Rational | [1, 2] | 0.7751 | ✓ tight | 0.3312 | ✓ tight |
| 11 | 325 | p=2 (100%) | Rational | [0, 1] | 0.6833 | ✓ tight | 0.6552 | ✓ tight |
| 12 | 391 | p=3 (30%) | Rational | [0, 2] | 0.8485 | ✗ loose | 0.5419 | ✓ tight |
| 13 | 50 | p=7 (30%) | Rational | [1, 1] | 0.8231 | ✗ loose | 0.5651 | ✓ tight |
| 14 | 496 | p=11 (28%) | Algebraic | [0, 1] | 0.8110 | ✗ loose | 0.6136 | ✓ tight |
| 15 | 294 | p=5 (31%) | Rational | [2, 2] | 0.8163 | ✗ loose | 0.4985 | ✓ tight |

**Within-code distance tighter than cross-code baseline**: 8/16 codes
**Conditional coherence tighter than unconditional baseline**: 16/16 codes
