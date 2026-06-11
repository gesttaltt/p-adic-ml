"""
eval_hierarchical_alignment.py

Computes per-prime metric alignment loss and Spearman r for the hierarchical
VQ-VAE's bottom-level quantized representations, compared to the flat Euclidean
and Poincaré hd=256 models.

The bottom codes z_q_bot are the fine-grained latent vectors; they are the
natural unit for alignment comparison since they carry the full per-sequence
representation after top-down conditioning.

Outputs:
  plots/hierarchical_alignment.md  — per-prime comparison table
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

PRIMES   = [2, 3, 5, 7, 11]
VOCAB    = max(PRIMES) + 2
N        = 64
SAMPLES  = 300
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS = {
    'Euc hd=64':    ('euclidean', './checkpoints/euclidean_n64/beta_vae_metric.pt',  64),
    'Hyp-P hd=256': ('poincare',  './checkpoints/hyperbolic_n64_hd256/hyperbolic_vae.pt', 256),
    'Hier hd=64':   ('hier',      './checkpoints/hierarchical/vqvae.pt',             64),
    'Hier hd=256':  ('hier',      './checkpoints/hierarchical_hd256/vqvae.pt',       256),
    'Hier B19 hd=64': ('hier19',  './checkpoints/hierarchical_broad19/vqvae.pt',     64),
}


def load_model(kind, path, hd):
    if kind == 'euclidean':
        m = ConditionalBetaVAE(vocab_size=VOCAB, hidden_dim=hd, latent_dim=32, N=N)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        return m.to(DEVICE).eval()
    elif kind == 'poincare':
        m = HyperbolicBetaVAE(vocab_size=VOCAB, hidden_dim=hd, latent_dim=32, N=N,
                              manifold='poincare', curvature=1.0)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        return m.to(DEVICE).eval()
    elif kind == 'hier':
        m = HierarchicalVQVAE(vocab_size=VOCAB, hidden_dim=hd, N=N)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        return m.to(DEVICE).eval()
    elif kind == 'hier19':
        vocab19 = 21   # max(primes_broad19)+2 = 19+2
        m = HierarchicalVQVAE(vocab_size=vocab19, hidden_dim=hd, N=N)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        return m.to(DEVICE).eval()


@torch.no_grad()
def get_latent(model, kind, digits, p_t):
    if kind == 'euclidean':
        mu, _ = model.encode(digits, p_t)
        return mu
    elif kind == 'poincare':
        mu_tang, _ = model.encode(digits, p_t)
        scale = 1.0 / math.sqrt(model.latent_dim)
        return model.manifold.expmap0(mu_tang * scale)
    elif kind in ('hier', 'hier19'):
        z_q_bot, _, _, _, _, _ = model.encode(digits, p_t)
        # z_q_bot is [B, L_bot, bot_dim]; flatten to [B, L_bot * bot_dim]
        B = z_q_bot.shape[0]
        return z_q_bot.reshape(B, -1)


def padic_latent_metrics(model, kind, p_val, dataset):
    seqs = [s for s in dataset if s['p'] == p_val and s['type'] != 2]
    if len(seqs) < 10:
        return None

    digits = torch.stack([s['digits'] for s in seqs]).to(DEVICE)
    p_t    = torch.full((len(seqs),), p_val, dtype=torch.long, device=DEVICE)

    z = get_latent(model, kind, digits, p_t)

    # alignment loss (Euclidean distance in latent space)
    loss = compute_metric_loss(z, digits, p_t).item()

    # Spearman r between pairwise p-adic and latent distances
    d_pad = batch_padic_distance(digits, p_t).cpu().numpy()
    diff  = z.unsqueeze(1) - z.unsqueeze(0)
    d_lat = diff.norm(dim=-1).cpu().numpy()
    idx   = np.triu_indices(len(seqs), k=1)
    r, _  = spearmanr(d_pad[idx], d_lat[idx])

    return {'loss': loss, 'spearman': float(r)}


def weighted_avg(results, key):
    total_w = total_v = 0.0
    for p, v in results.items():
        if v and not math.isnan(v[key]):
            w = math.log(p) + 1
            total_w += w; total_v += w * v[key]
    return total_v / total_w if total_w else float('nan')


def main():
    print(f'Device: {DEVICE}')
    ds = PadicDataset(primes=PRIMES, N=N, num_samples_per_type=SAMPLES)

    all_results = {}
    for name, (kind, path, hd) in MODELS.items():
        print(f'\nEvaluating {name}...')
        model = load_model(kind, path, hd)
        per_prime = {}
        for p in PRIMES:
            per_prime[p] = padic_latent_metrics(model, kind, p, ds)
        all_results[name] = per_prime
        del model

    # Print table
    header = f"{'Prime':<8}" + ''.join(f"  {n:<22}" for n in MODELS)
    print('\n' + header)
    print('-' * len(header))
    for p in PRIMES:
        row = f'p={p:<6}'
        for name in MODELS:
            v = all_results[name].get(p)
            if v:
                row += f"  loss={v['loss']:.5f} r={v['spearman']:.4f} "
            else:
                row += f"  {'—':<22}"
        print(row)

    # Save markdown
    lines = ['# Hierarchical VQ-VAE — Metric Alignment Comparison\n',
             f'N={N}, {SAMPLES} samples/type/prime, Broad-11 eval set\n',
             '| Prime | ' + ' | '.join(f'{n} Loss / $r$' for n in MODELS) + ' |',
             '| :---: | ' + ' | '.join([':---:'] * len(MODELS)) + ' |']
    for p in PRIMES:
        row = f'| $p={p}$ |'
        for name in MODELS:
            v = all_results[name].get(p)
            row += f" {v['loss']:.5f} / {v['spearman']:.4f} |" if v else ' — |'
        lines.append(row)

    lines.append('| **Wtd avg** |')
    wtd_row = '| **Wtd avg** |'
    for name in MODELS:
        wl = weighted_avg(all_results[name], 'loss')
        wr = weighted_avg(all_results[name], 'spearman')
        wtd_row += f' **{wl:.5f}** / **{wr:.4f}** |'
    lines[-1] = wtd_row

    content = '\n'.join(lines)
    os.makedirs('./plots', exist_ok=True)
    with open('./plots/hierarchical_alignment.md', 'w') as f:
        f.write(content + '\n')
    print('\n\n' + content)
    print('\nSaved to ./plots/hierarchical_alignment.md')


if __name__ == '__main__':
    main()
