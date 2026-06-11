"""
analyze_cross_prime.py

Quantitative analysis of cross-prime latent interpolation paths.

For each (p_start, p_end) pair and each model, we draw NUM_PAIRS random
(seq_start, seq_end) pairs, linearly interpolate z_start → z_end, and at
each step decode with p_end. We record per-step:

  digit_entropy      — Shannon entropy of the decoded digit distribution over
                       positions 0..N-1. High = uniform/confused, low = structured.
  dist_to_start      — mean p-adic distance between z(t) decoded and z(0) decoded
  dist_to_end        — mean p-adic distance between z(t) decoded and z(1) decoded

A monotonic dist_to_start↑ / dist_to_end↓ transition indicates the model
learned a topologically meaningful cross-base path. A step-function jump
indicates a discontinuous, prime-partitioned space.

Outputs:
  plots/cross_prime_analysis.png       — 2×3 subplot grid (model × prime-pair)
  plots/cross_prime_analysis.md        — markdown summary table
"""
import sys, os; _r = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(_r, 'src')); os.chdir(_r)

import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import PadicDataset
from beta_vae import ConditionalBetaVAE
from hyperbolic_vae import HyperbolicBetaVAE
from metric_alignment import batch_padic_distance

# ── config ────────────────────────────────────────────────────────────────────
PRIME_PAIRS  = [(2, 5), (2, 11), (5, 11)]
N            = 64
NUM_PAIRS    = 60      # interpolation pairs averaged per (model, prime-pair)
NUM_STEPS    = 11
SAMPLES_PER  = 200     # sequences loaded per prime for endpoint sampling
VOCAB        = 13      # Broad-11 vocab
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS = {
    'Euclidean hd=64': {
        'ckpt': './checkpoints/euclidean_n64/beta_vae_metric.pt',
        'cls':  'euclidean',
        'hd':   64,
    },
    'Poincaré hd=256': {
        'ckpt': './checkpoints/hyperbolic_n64_hd256/hyperbolic_vae.pt',
        'cls':  'poincare',
        'hd':   256,
    },
}

# ── model loading ─────────────────────────────────────────────────────────────

def load_model(cfg):
    if cfg['cls'] == 'euclidean':
        m = ConditionalBetaVAE(vocab_size=VOCAB, hidden_dim=cfg['hd'], latent_dim=32, N=N)
    else:
        m = HyperbolicBetaVAE(vocab_size=VOCAB, hidden_dim=cfg['hd'], latent_dim=32, N=N,
                              manifold=cfg['cls'], curvature=1.0)
    m.load_state_dict(torch.load(cfg['ckpt'], map_location=DEVICE))
    return m.to(DEVICE).eval()

# ── encoding / decoding helpers ───────────────────────────────────────────────

@torch.no_grad()
def encode_mu(model, digits, p_val, is_hyp):
    p_t = torch.tensor([p_val], dtype=torch.long, device=DEVICE)
    d_t = digits.unsqueeze(0).to(DEVICE)
    if is_hyp:
        mu_tang, _ = model.encode(d_t, p_t)
        scale = 1.0 / math.sqrt(model.latent_dim)
        return model.manifold.expmap0(mu_tang * scale)  # on manifold
    else:
        mu, _ = model.encode(d_t, p_t)
        return mu  # Euclidean vector

@torch.no_grad()
def interp_and_decode(model, z1, z2, t, p_decode, is_hyp):
    p_t = torch.tensor([p_decode], dtype=torch.long, device=DEVICE)
    if is_hyp:
        v   = model.manifold.logmap(z1, z2)
        z_t = model.manifold.projx(model.manifold.expmap(z1, t * v))
    else:
        z_t = (1 - t) * z1 + t * z2
    logits  = model.decode(z_t, p_t)
    decoded = torch.argmax(logits, dim=-1)[0].cpu()
    return decoded

# ── per-step statistics ───────────────────────────────────────────────────────

def digit_entropy(seq, p):
    """Shannon entropy of the digit-frequency distribution (nats)."""
    counts = np.bincount(seq, minlength=p).astype(float)
    counts += 1e-12
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))

def padic_dist(seq_a, seq_b, p):
    """p-adic distance between two sequences (scalar)."""
    for i, (a, b) in enumerate(zip(seq_a, seq_b)):
        if a != b:
            return float(p ** -i)
    return 0.0

# ── main analysis ─────────────────────────────────────────────────────────────

def analyze_pair(model, is_hyp, ds_start, ds_end, p_start, p_end):
    """
    Returns arrays of shape [NUM_STEPS] averaged over NUM_PAIRS pairs.
    """
    seqs_start = [s['digits'] for s in ds_start if s['type'] != 2]
    seqs_end   = [s['digits'] for s in ds_end   if s['type'] != 2]

    t_vals = np.linspace(0, 1, NUM_STEPS)
    all_entropy   = np.zeros((NUM_PAIRS, NUM_STEPS))
    all_dist_s    = np.zeros((NUM_PAIRS, NUM_STEPS))
    all_dist_e    = np.zeros((NUM_PAIRS, NUM_STEPS))

    rng = np.random.default_rng(42)
    for k in range(NUM_PAIRS):
        s_idx = rng.integers(len(seqs_start))
        e_idx = rng.integers(len(seqs_end))
        z1 = encode_mu(model, seqs_start[s_idx], p_start, is_hyp)
        z2 = encode_mu(model, seqs_end[e_idx],   p_end,   is_hyp)

        path_seqs = []
        for t in t_vals:
            dec = interp_and_decode(model, z1, z2, float(t), p_end, is_hyp)
            path_seqs.append(dec.numpy().tolist())

        seq0 = path_seqs[0]
        seq1 = path_seqs[-1]

        for step, (t, seq) in enumerate(zip(t_vals, path_seqs)):
            all_entropy[k, step] = digit_entropy(seq, p_end)
            all_dist_s[k, step]  = padic_dist(seq, seq0, p_end)
            all_dist_e[k, step]  = padic_dist(seq, seq1, p_end)

    return {
        't':            t_vals,
        'entropy':      all_entropy.mean(axis=0),
        'entropy_std':  all_entropy.std(axis=0),
        'dist_to_start': all_dist_s.mean(axis=0),
        'dist_to_end':   all_dist_e.mean(axis=0),
    }

# ── plotting ──────────────────────────────────────────────────────────────────

def make_plot(results, save_path):
    """
    results: dict[model_name][prime_pair_str] -> stats dict
    """
    model_names = list(results.keys())
    n_rows = len(model_names)
    n_cols = len(PRIME_PAIRS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=130)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r, mname in enumerate(model_names):
        for c, (ps, pe) in enumerate(PRIME_PAIRS):
            key  = f'{ps}→{pe}'
            stat = results[mname][key]
            t    = stat['t']
            ax   = axes[r, c]

            ax2 = ax.twinx()

            l1, = ax.plot(t, stat['dist_to_start'], 'b-o', ms=4, label='dist to start')
            l2, = ax.plot(t, stat['dist_to_end'],   'r-o', ms=4, label='dist to end')
            l3, = ax2.plot(t, stat['entropy'],       'g--s', ms=4, label='digit entropy')

            ax.fill_between(t,
                stat['dist_to_start'] - 0.5 * stat.get('entropy_std', 0),
                stat['dist_to_start'] + 0.5 * stat.get('entropy_std', 0),
                alpha=0.08, color='blue')

            ax.set_xlabel('t')
            ax.set_ylabel('p-adic distance', color='black')
            ax2.set_ylabel('entropy (nats)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            ax.set_title(f'{mname}\n$p={ps} \\to p={pe}$', fontsize=10)
            ax.grid(True, alpha=0.3)

            if r == 0 and c == n_cols - 1:
                lines = [l1, l2, l3]
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, fontsize=8, loc='upper left')

    plt.suptitle('Cross-Prime Interpolation Analysis\n'
                 'dist to start (blue) / dist to end (red) / digit entropy (green)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to {save_path}')

# ── markdown report ───────────────────────────────────────────────────────────

def make_report(results, save_path):
    lines = ['# Cross-Prime Interpolation Analysis\n',
             f'Models: {list(results.keys())}  ',
             f'Prime pairs: {PRIME_PAIRS}  ',
             f'Pairs averaged: {NUM_PAIRS}, Steps: {NUM_STEPS}\n']

    for mname, pair_results in results.items():
        lines.append(f'\n## {mname}\n')
        lines.append('| Prime pair | Entropy t=0 | Entropy t=0.5 | Entropy t=1 | '
                     'Dist-start t=0.5 | Dist-end t=0.5 | Monotone? |')
        lines.append('| :--- | :---: | :---: | :---: | :---: | :---: | :---: |')
        for ps, pe in PRIME_PAIRS:
            key  = f'{ps}→{pe}'
            stat = pair_results[key]
            mid  = NUM_STEPS // 2
            ent  = stat['entropy']
            ds   = stat['dist_to_start']
            de   = stat['dist_to_end']
            # monotone: dist_to_start should be non-decreasing, dist_to_end non-increasing
            mono_s = bool(np.all(np.diff(ds) >= -1e-6))
            mono_e = bool(np.all(np.diff(de) <=  1e-6))
            mono   = 'Yes' if (mono_s and mono_e) else 'Partial' if (mono_s or mono_e) else 'No'
            lines.append(
                f'| $p={ps} \\to p={pe}$ '
                f'| {ent[0]:.3f} | {ent[mid]:.3f} | {ent[-1]:.3f} '
                f'| {ds[mid]:.4f} | {de[mid]:.4f} | {mono} |'
            )

    content = '\n'.join(lines)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(content + '\n')
    print(f'Saved report to {save_path}')
    return content

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print(f'Device: {DEVICE}')
    print(f'Prime pairs: {PRIME_PAIRS}')
    print(f'Pairs per combination: {NUM_PAIRS}\n')

    # Pre-load datasets (one per unique prime)
    all_primes = sorted({p for pair in PRIME_PAIRS for p in pair})
    print('Loading datasets...')
    datasets = {}
    for p in all_primes:
        datasets[p] = PadicDataset(primes=[p], N=N, num_samples_per_type=SAMPLES_PER)
        print(f'  p={p}: {len(datasets[p])} sequences')

    results = {}
    for mname, cfg in MODELS.items():
        print(f'\n{"="*60}')
        print(f'Model: {mname}')
        print(f'{"="*60}')
        model  = load_model(cfg)
        is_hyp = cfg['cls'] != 'euclidean'
        results[mname] = {}

        for p_start, p_end in PRIME_PAIRS:
            print(f'  Analyzing p={p_start} → p={p_end} ({NUM_PAIRS} pairs)...')
            stat = analyze_pair(model, is_hyp,
                                datasets[p_start], datasets[p_end],
                                p_start, p_end)
            results[mname][f'{p_start}→{p_end}'] = stat

            # Quick summary
            t   = stat['t']
            mid = NUM_STEPS // 2
            print(f'    Entropy:    t=0 {stat["entropy"][0]:.3f}  '
                  f't=0.5 {stat["entropy"][mid]:.3f}  t=1 {stat["entropy"][-1]:.3f}')
            print(f'    Dist→start: t=0 {stat["dist_to_start"][0]:.4f}  '
                  f't=0.5 {stat["dist_to_start"][mid]:.4f}  t=1 {stat["dist_to_start"][-1]:.4f}')
            print(f'    Dist→end:   t=0 {stat["dist_to_end"][0]:.4f}  '
                  f't=0.5 {stat["dist_to_end"][mid]:.4f}  t=1 {stat["dist_to_end"][-1]:.4f}')

        del model  # free memory before next model

    make_plot(results,  './plots/cross_prime_analysis.png')
    report = make_report(results, './plots/cross_prime_analysis.md')
    print('\n' + report)


if __name__ == '__main__':
    main()
