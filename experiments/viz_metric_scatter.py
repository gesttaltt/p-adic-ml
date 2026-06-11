import sys, os; _r = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(_r, 'src')); os.chdir(_r)
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from dataset import PadicDataset
from beta_vae import ConditionalBetaVAE
from metric_alignment import batch_padic_distance
from viz_style import apply_style, prime_color

apply_style()


def _load_model(path, vocab_size, hidden_dim, latent_dim, N, device):
    m = ConditionalBetaVAE(
        vocab_size=vocab_size, hidden_dim=hidden_dim,
        latent_dim=latent_dim, N=N,
    )
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    return m


def _encode_batch(model, digits_t, p_t):
    with torch.no_grad():
        mu, lv = model.encode(digits_t, p_t)
        return model.reparameterize(mu, lv)


def _pairwise_euclidean(z):
    """Upper-triangle pairwise Euclidean distances as a flat numpy array."""
    z1 = z.unsqueeze(1)
    z2 = z.unsqueeze(0)
    D = torch.norm(z1 - z2, p=2, dim=-1).cpu().numpy()
    idx = np.triu_indices(D.shape[0], k=1)
    return D[idx]


def run_metric_scatter(
    model_paths,
    model_labels,
    primes=(2, 3, 5, 7, 11),
    N=64,
    vocab_size=13,
    hidden_dim=64,
    latent_dim=32,
    n_seq=80,
    save_path='./plots/metric_scatter.png',
    device='cpu',
):
    n_models = len(model_paths)
    n_primes = len(primes)

    models = [
        _load_model(p, vocab_size, hidden_dim, latent_dim, N, device)
        for p in model_paths
    ]

    fig, axes = plt.subplots(
        n_primes, n_models,
        figsize=(4.5 * n_models, 4.0 * n_primes),
        squeeze=False,
    )
    fig.suptitle('p-adic distance vs latent Euclidean distance', fontsize=14)

    for row, p in enumerate(primes):
        ds = PadicDataset(primes=[p], N=N, num_samples_per_type=n_seq)
        samples = [s for s in ds if s['type'] in (0, 1)][:n_seq]
        digits_t = torch.stack([s['digits'] for s in samples]).to(device)
        p_t = torch.full((len(samples),), p, dtype=torch.long, device=device)

        # p-adic distances are the same for all models
        D_padic_full = batch_padic_distance(digits_t, p_t).cpu().numpy()
        idx = np.triu_indices(D_padic_full.shape[0], k=1)
        d_padic = D_padic_full[idx]

        for col, (model, label) in enumerate(zip(models, model_labels)):
            ax = axes[row][col]
            z = _encode_batch(model, digits_t, p_t)
            d_latent = _pairwise_euclidean(z)

            color = prime_color(p)
            ax.scatter(d_padic, d_latent, color=color, s=6, alpha=0.35, linewidths=0)

            # Reference line: OLS through origin (slope = cov/var)
            slope = (d_padic * d_latent).sum() / (d_padic ** 2).sum()
            x_line = np.array([0, d_padic.max()])
            ax.plot(x_line, slope * x_line, color='#333333',
                    linestyle='--', linewidth=1.2, label='ideal fit')

            r, _ = spearmanr(d_padic, d_latent)
            ax.set_title(f'{label}  p={p}', fontsize=10)
            ax.set_xlabel('d_padic', fontsize=9)
            ax.set_ylabel('d_latent', fontsize=9)
            ax.text(0.97, 0.05, f'Spearman r = {r:.3f}',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=9, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f'Saved → {save_path}')


def _parse_args():
    ap = argparse.ArgumentParser(description='Metric alignment scatter plot')
    ap.add_argument('--model_paths', nargs='+', required=True,
                    help='One or more checkpoint paths')
    ap.add_argument('--model_labels', nargs='+',
                    help='Display labels matching --model_paths')
    ap.add_argument('--primes',     nargs='+', type=int, default=[2, 3, 5, 7, 11])
    ap.add_argument('--N',          type=int, default=64)
    ap.add_argument('--vocab_size', type=int, default=13)
    ap.add_argument('--hidden_dim', type=int, default=64)
    ap.add_argument('--latent_dim', type=int, default=32)
    ap.add_argument('--n_seq',      type=int, default=80,
                    help='Sequences per prime (→ n_seq*(n_seq-1)/2 pairs)')
    ap.add_argument('--save_path',  default='./plots/metric_scatter.png')
    return ap.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    labels = args.model_labels or [os.path.basename(p) for p in args.model_paths]
    if len(labels) != len(args.model_paths):
        raise ValueError('--model_labels must have the same count as --model_paths')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_metric_scatter(
        model_paths=args.model_paths,
        model_labels=labels,
        primes=args.primes,
        N=args.N,
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_seq=args.n_seq,
        save_path=args.save_path,
        device=device,
    )
