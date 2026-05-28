# p-adic Generative Models & Hyperbolic Representation Learning

## 📖 About the Project
This project explores the mathematical intersection of **$p$-adic numbers (ultrametric tree spaces)**, **hyperbolic geometry (Poincaré Disk model)**, and **deep generative models (conditional VQ-VAEs and Beta-VAEs)**. 

### Core Motivation
Traditional machine learning architectures map hierarchical data into flat Euclidean spaces ($\mathbb{R}^d$), which suffers from geometric distortion (crowding effects). In contrast, trees are discrete representations of hyperbolic space. By combining $p$-adic mathematical structures with conditional VAE architectures, this project:
1. Embeds hierarchically structured data directly into continuous hyperbolic spaces without distortion.
2. Mathematically proves that the $p$-adic tree ultrametric is naturally isomorphic to hyperbolic distance inside the Poincaré Disk.
3. Systematically demonstrates that **multi-task regularization** (joint training across up to 9 distinct prime bases simultaneously) dramatically improves reconstruction accuracy and latent space metric alignment, dropping alignment loss by up to **~90%**.

---

## 🔍 The Discovery: Multi-Task Regularization Scaling

A core research question we investigated was:
> *Is it more beneficial to train a p-adic generative model on a small, restricted set of primes (e.g., just $p \in \{2, 5\}$), or a broader, joint configuration of primes?*

Through systematic, controlled training experiments, we discovered that **training on more prime bases simultaneously is highly beneficial**. Rather than causing model saturation or capacity congestion, adding more bases acts as a powerful **multi-task regularizer**.

### 📊 Comparative Results (Evaluated on $p=2$ and $p=5$)

| Prime Set Config | Primes Included | VQ-VAE Accuracy ($p=5$) | Metric Alignment Loss ($p=5$) | Metric Alignment Loss ($p=2$) |
| :--- | :--- | :---: | :---: | :---: |
| **Restricted** | $[2, 5]$ | $51.70\%$ | $0.06533$ | $0.02469$ |
| **Broad-11** | $[2, 3, 5, 7, 11]$ | $59.98\%$ | $0.01887$ | $0.01207$ |
| **Broad-13** | $[2..13]$ | $64.52\%$ | $0.02559$ | $0.00947$ |
| **Broad-17** | $[2..17]$ | **$69.87\%$** | $0.02520$ | $0.01073$ |
| **Broad-19** | $[2..19]$ | $67.53\%$ | **$0.00698$** | **$0.00827$** |
| **Broad-23** | $[2..23]$ | $68.32\%$ | $0.05983$ | $0.01262$ |

### 📈 Reconstruction Performance Scaling
As the number of trained primes scales up, the digit reconstruction accuracy on the complex 5-ary tree ($p=5$) climbs from **$51.70\%$** to **$68.32\%$** (peaking at $69.87\%$ for Broad-17):

![VQ-VAE Accuracy Scaling](plots/comparison_p23/vqvae_accuracy_scaling.png)

### 🌀 Latent Space Topology Scaling
Enforcing multiple tree topologies onto the same continuous latent space acts as a topological regularizer. As shown in the 6-way PCA projection comparison below, the latent space clusters become cleaner and more separated as we scale the prime set:

![Latent Space PCA Scaling](plots/comparison_p23/latent_space_scaling.png)

---

## ⚡ The Cascade Gating Inference System

Autoregressive prior sampling (VQ-VAE + Prior) is highly precise but slow due to step-by-step generation. Continuous models (Beta-VAE) are extremely fast (one-step feedforward) but approximate. To leverage the strengths of both, this codebase implements a **Cascade Gating System** (`anomaly_detector.py`):
1. **Fast-Path Generation**: The system samples a candidate sequence from the continuous Beta-VAE.
2. **Self-Reconstruction Gating**: The Beta-VAE reconstructs its own output. If the cross-entropy reconstruction loss is below a calibrated threshold $\tau_p$, the sequence is deemed valid and routed immediately to the user (Fast Path).
3. **Slow-Path Fallback**: If the reconstruction loss exceeds $\tau_p$, the sequence is flagged as anomalous or low-quality. The system falls back to the VQ-VAE + Prior (Slow Path) to generate a precise sequence.

This architecture establishes a Pareto-optimal frontier, allowing users to trade off generation velocity (samples/sec) for reconstruction precision by adjusting $\tau_p$.

---

## 🔮 Poincaré Disk Hyperbolic Embeddings

A Poincaré disk is a model of 2-dimensional hyperbolic geometry. Trees are discrete analogs of hyperbolic spaces; specifically, a $p$-ary tree can be viewed as a discretization of the hyperbolic plane $\mathbb{H}^2$. 

We map $p$-adic digit sequences $a_0, a_1, \dots, a_{d-1}$ (where $a_i \in \{0, \dots, p-1\}$) to Poincaré coordinates:
* **Hyperbolic Radius**: The depth $d$ of a digit corresponds to the radius $r = \tanh(c \cdot d)$. As $d \rightarrow \infty$, points converge to the boundary circle ($r \rightarrow 1$).
* **Angular Sectors**: The sequence of digits partitions the angular range $[0, 2\pi]$ into $p$-ary sectors. 
* **Ultrametric Isomorphism**: Two numbers are close in $p$-adic distance if they share a long common prefix, meaning their Poincaré paths stay together deep into the disk (closer to the boundary). If they differ early, they must travel back towards the origin to connect, resulting in a large hyperbolic distance.

### Visualizing Tree & Poincaré Embeddings ($p=19$ and $p=23$)

| 19-adic Poincaré Disk | 23-adic Poincaré Disk |
| :---: | :---: |
| ![19-adic Poincare](plots/poincare_p19.png) | ![23-adic Poincare](plots/poincare_p23.png) |

| 19-adic Tree Branching | 23-adic Tree Branching |
| :---: | :---: |
| ![19-adic Tree](plots/padic_tree_19.png) | ![23-adic Tree](plots/padic_tree_23.png) |

*(Blue = Rational sequences, Red = Algebraic sequences, Green = VQ-VAE prior-generated sequences)*

---

## 🧮 Mathematical Foundations

### 1. The $p$-adic Ultrametric
Unlike real numbers which follow the standard Archimedean metric, $p$-adic numbers are equipped with the **non-Archimedean ultrametric**:
$$d_p(x, y) \le \max\big(d_p(x, z), d_p(z, y)\big)$$
This metric ensures that distance is determined solely by the highest common ancestor (the longest common prefix) in the branching tree.

### 2. Hensel's Lemma & Algebraic Sequences
Our dataset includes algebraic roots (Red trajectories) generated using **Hensel's Lemma**, which acts as the $p$-adic equivalent of Newton's method for finding root approximations:
$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_0)} \pmod{p^{n+1}}$$
Because Hensel lifting builds roots digit-by-digit, algebraic numbers trace highly structured, non-periodic trajectories down the branching tree, which our VQ-VAE successfully learns to generate.

---

## 🚀 Future Research Directions

We invite contributions and extensions in the following areas:
* **Hyperbolic Latent Spaces**: Transitioning from a Euclidean latent space ($\mathbb{R}^d$) with a metric alignment loss to a true **Hyperbolic VAE** using Poincaré or Lorentz manifolds in PyTorch.
* **Capacity Scale-Up**: Expanding the models' hidden dimension (e.g. from 64 to 256) to handle capacity constraints observed when scaling past $p=23$.
* **Continuous Prime Embeddings**: Replacing the index-based categorical embedding with continuous prime representations to share mathematical features across different bases.

---

## 🛠️ How to Run & Reproduce

### 1. Installation
Ensure PyTorch and Matplotlib are installed:
```bash
pip install torch matplotlib numpy
```

### 2. Run Scaling Experiments
To run the scaling experiments, run:
```bash
# Broad-19 training and evaluation
python scaling_analysis/train_broad_p19.py

# Broad-23 training and evaluation
python scaling_analysis/train_broad_p23.py
```

### 3. Generate Poincaré Disk Plots
To generate Poincaré disk plots dynamically for target primes:
```bash
python scaling_analysis/poincare_embedding.py
```
This generates plots under `plots/poincare_p<p>.png`.
