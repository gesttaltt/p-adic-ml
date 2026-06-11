"""
eval_algebraic_only.py  —  Item #31

Re-runs the metric alignment evaluation filtering to algebraic sequences only
(seq_type == 1: Hensel-lifted polynomial roots). These are the most structurally
challenging sequences — they follow precise non-periodic trajectories down the
p-adic tree, determined by the root-lifting polynomial.

Comparing algebraic-only vs mixed (rational + algebraic) results reveals whether
models that perform well on average are genuinely capturing p-adic ultrametric
structure or partly benefiting from the regularity of rational sequences.

Models evaluated (same as eval_hyperbolic_hd256.py):
  - Euclidean hd=64
  - Poincaré hd=256, c=1.0
  - Poincaré hd=256, c=5.0  (alignment-optimal)
  - Hierarchical hd=64

Output: plots/algebraic_alignment.md
"""
import sys, os; _r = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(_r, 'src')); os.chdir(_r)

import os, math, torch
import numpy as np
from scipy.stats import spearmanr

from dataset import PadicDataset
from beta_vae import ConditionalBetaVAE
from hyperbolic_vae import HyperbolicBetaVAE
from hierarchical_vqvae import HierarchicalVQVAE
from metric_alignment import batch_padic_distance, compute_metric_loss

PRIMES  = [2, 3, 5, 7, 11]
VOCAB   = max(PRIMES) + 2
N       = 64
SAMPLES = 300   # per type per prime (we'll filter to algebraic only)
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS = {
    'Euc hd=64':     ('euclidean', './checkpoints/euclidean_n64/beta_vae_metric.pt',      64,  None),
    'Hyp-P hd=256 c=1': ('poincare', './checkpoints/hyperbolic_n64_hd256/hyperbolic_vae.pt', 256, 1.0),
    'Hyp-P hd=256 c=5': ('poincare', './checkpoints/hyperbolic_n64_hd256_c5/hyperbolic_vae.pt', 256, 5.0),
    'Hier hd=64':    ('hier',      './checkpoints/hierarchical/vqvae.pt',               64,  None),
}

# Reference numbers from full-dataset evaluation (eval_hyperbolic_hd256.py)
MIXED_REF = {
    'Euc hd=64':        {2: (0.00891, 0.9116), 3: (0.01140, 0.8131), 5: (0.03150, 0.6894),
                          7: (0.04935, 0.5999), 11: (0.06286, 0.4515)},
    'Hyp-P hd=256 c=1': {2: (0.00643, 0.9201), 3: (0.01029, 0.8161), 5: (0.01020, 0.6903),
                          7: (0.00977, 0.6039), 11: (0.01145, 0.4957)},
    'Hyp-P hd=256 c=5': {2: (0.01205, 0.9257), 3: (0.00857, 0.8310), 5: (0.00534, 0.6962),
                          7: (0.00300, 0.6085), 11: (0.00346, 0.4962)},
    'Hier hd=64':       {2: (0.28296, 0.0895), 3: (0.23256, 0.0579), 5: (0.16257, 0.0459),
                          7: (0.12405, 0.0377), 11: (0.08579, 0.0322)},
}


def load_model(kind, path, hd, curvature):
    if kind == 'euclidean':
        m = ConditionalBetaVAE(vocab_size=VOCAB, hidden_dim=hd, latent_dim=32, N=N)
    elif kind == 'poincare':
        m = HyperbolicBetaVAE(vocab_size=VOCAB, hidden_dim=hd, latent_dim=32, N=N,
                              manifold='poincare', curvature=curvature)
    elif kind == 'hier':
        m = HierarchicalVQVAE(vocab_size=VOCAB, hidden_dim=hd, N=N)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.to(DEVICE).eval()


@torch.no_grad()
def get_latent(model, kind, digits, p_t):
    if kind == 'euclidean':
        mu, _ = model.encode(digits, p_t)
        return mu
    elif kind == 'poincare':
        mu_tang, _ = model.encode(digits, p_t)
        return model.manifold.expmap0(mu_tang / math.sqrt(model.latent_dim))
    elif kind == 'hier':
        z_q_bot, _, _, _, _, _ = model.encode(digits, p_t)
        return z_q_bot.reshape(z_q_bot.shape[0], -1)


def eval_per_prime_algebraic(model, kind, p_val, dataset):
    """Evaluate only on algebraic sequences (seq_type == 1)."""
    seqs = [s for s in dataset if s['p'] == p_val and s['type'] == 1]
    if len(seqs) < 10:
        return None

    digits = torch.stack([s['digits'] for s in seqs]).to(DEVICE)
    p_t    = torch.full((len(seqs),), p_val, dtype=torch.long, device=DEVICE)
    z      = get_latent(model, kind, digits, p_t)

    loss   = compute_metric_loss(z, digits, p_t).item()

    d_pad  = batch_padic_distance(digits, p_t).cpu().numpy()
    diff   = z.unsqueeze(1) - z.unsqueeze(0)
    d_lat  = diff.norm(dim=-1).cpu().numpy()
    idx    = np.triu_indices(len(seqs), k=1)
    r, _   = spearmanr(d_pad[idx], d_lat[idx])
    return {'loss': loss, 'spearman': float(r), 'n': len(seqs)}


def weighted_avg(results, key):
    tw = tv = 0.0
    for p, v in results.items():
        if v and not math.isnan(v[key]):
            w = math.log(p) + 1; tw += w; tv += w * v[key]
    return tv / tw if tw else float('nan')


def main():
    print(f'Device: {DEVICE}')
    ds = PadicDataset(primes=PRIMES, N=N, num_samples_per_type=SAMPLES)
    n_alg = sum(1 for s in ds if s['type'] == 1)
    print(f'Algebraic sequences: {n_alg} / {len(ds)} total')

    all_results = {}
    for name, (kind, path, hd, curv) in MODELS.items():
        print(f'\nEvaluating {name} (algebraic only)...')
        model = load_model(kind, path, hd, curv)
        per_prime = {p: eval_per_prime_algebraic(model, kind, p, ds) for p in PRIMES}
        all_results[name] = per_prime
        del model

    # Print comparison
    print(f'\n{"Prime":<8}', end='')
    for name in MODELS:
        print(f'  {name:<26}', end='')
    print()
    print('-' * (8 + 28 * len(MODELS)))

    for p in PRIMES:
        print(f'p={p:<6}', end='')
        for name in MODELS:
            v = all_results[name].get(p)
            if v:
                print(f'  loss={v["loss"]:.5f} r={v["spearman"]:.4f}    ', end='')
            else:
                print(f'  {"—":<26}', end='')
        print()

    # Markdown report
    lines = ['# Algebraic-Only Metric Alignment Evaluation\n',
             f'seq_type=1 (Hensel-lifted roots) only | N={N} | {SAMPLES} samples/type/prime\n',
             '## Per-Prime Results\n',
             '| Prime | ' + ' | '.join(f'{n} Loss / $r$' for n in MODELS) + ' |',
             '| :---: | ' + ' | '.join([':---:'] * len(MODELS)) + ' |']

    for p in PRIMES:
        row = f'| $p={p}$ |'
        for name in MODELS:
            v = all_results[name].get(p)
            row += f' {v["loss"]:.5f} / {v["spearman"]:.4f} |' if v else ' — |'
        lines.append(row)

    # Weighted avg row
    row = '| **Wtd avg** |'
    for name in MODELS:
        wl = weighted_avg(all_results[name], 'loss')
        wr = weighted_avg(all_results[name], 'spearman')
        row += f' **{wl:.5f}** / **{wr:.4f}** |'
    lines.append(row)

    lines += ['\n## Algebraic vs Mixed — Change in Spearman $r$ (wtd avg)\n',
              '| Model | Mixed $r$ (ref) | Algebraic $r$ | Change |',
              '| :--- | :---: | :---: | :---: |']
    for name in MODELS:
        alg_r = weighted_avg(all_results[name], 'spearman')
        ref   = MIXED_REF.get(name, {})
        if ref:
            mix_r = sum((math.log(p)+1)*ref[p][1] for p in PRIMES) / \
                    sum(math.log(p)+1 for p in PRIMES)
            delta = alg_r - mix_r
            flag  = '↑' if delta > 0.01 else ('↓' if delta < -0.01 else '≈')
            lines.append(f'| {name} | {mix_r:.4f} | {alg_r:.4f} | {delta:+.4f} {flag} |')

    content = '\n'.join(lines)
    os.makedirs('./plots', exist_ok=True)
    with open('./plots/algebraic_alignment.md', 'w') as f:
        f.write(content + '\n')
    print(f'\nSaved to ./plots/algebraic_alignment.md')
    print('\n' + content)


if __name__ == '__main__':
    main()
