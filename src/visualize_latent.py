import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dataset import PadicDataset
from beta_vae import ConditionalBetaVAE
from viz_style import (apply_style, RATIONAL_COLOR, ALGEBRAIC_COLOR,
                       RANDOM_COLOR, seq_type_legend_handles)

apply_style()

# Marker style per sequence type: rational=circle, algebraic=triangle
_TYPE_MARKER = {0: 'o', 1: '^', 2: 's'}
_TYPE_COLOR  = {0: RATIONAL_COLOR, 1: ALGEBRAIC_COLOR, 2: RANDOM_COLOR}
_TYPE_LABEL  = {0: 'Rational', 1: 'Algebraic', 2: 'Random'}


def project_pca(z, num_components=2):
    """
    PCA projection via torch SVD.

    Parameters
    ----------
    z : [B, D] tensor
    Returns : [B, num_components] projected coordinates
    """
    z_mean = torch.mean(z, dim=0, keepdim=True)
    z_centered = z - z_mean
    _, _, Vh = torch.linalg.svd(z_centered, full_matrices=False)
    return torch.matmul(z_centered, Vh[:num_components].t())


def evaluate_and_plot_latents(
    unaligned_path,
    aligned_path,
    save_dir='./plots',
    N=64,
    vocab_size=13,
    hidden_dim=64,
    latent_dim=32,
    primes=None,
    num_samples=150,
):
    """
    PCA scatter comparison: unaligned vs metric-aligned Beta-VAE latent spaces.

    Each subplot uses marker shape (○ rational, △ algebraic) and colour by
    p-adic residue (a₀ + a₁·p) for fine-grained cluster structure.
    """
    if primes is None:
        primes = [2, 3, 5, 7, 11]
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _load(path):
        m = ConditionalBetaVAE(
            vocab_size=vocab_size, hidden_dim=hidden_dim,
            latent_dim=latent_dim, N=N,
        )
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device).eval()
        return m

    model_unaligned = _load(unaligned_path)
    model_aligned   = _load(aligned_path)

    for p in primes:
        print(f'  p={p} …', end=' ', flush=True)
        ds = PadicDataset(primes=[p], N=N, num_samples_per_type=num_samples)

        digits_list, types, residues = [], [], []
        for sample in ds:
            if sample['type'] == 2:   # skip random noise for cleaner topology
                continue
            digits_list.append(sample['digits'])
            types.append(sample['type'].item())
            residues.append(sample['digits'][0].item() + sample['digits'][1].item() * p)

        digits_t = torch.stack(digits_list).to(device)
        p_t = torch.full((len(digits_list),), p, dtype=torch.long, device=device)
        types = np.array(types)
        residues = np.array(residues)

        with torch.no_grad():
            mu_un, lv_un = model_unaligned.encode(digits_t, p_t)
            z_un = model_unaligned.reparameterize(mu_un, lv_un)

            mu_al, lv_al = model_aligned.encode(digits_t, p_t)
            z_al = model_aligned.reparameterize(mu_al, lv_al)

        z_un_2d = project_pca(z_un.cpu(), 2).numpy()
        z_al_2d = project_pca(z_al.cpu(), 2).numpy()

        fig, (ax_un, ax_al) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f'Latent Space Topology — {p}-adic  (residue mod {p}²)',
            fontsize=13, fontweight='bold',
        )

        for ax, z2d, title in [
            (ax_un, z_un_2d, 'Standard Beta-VAE'),
            (ax_al, z_al_2d, 'Metric-Aligned Beta-VAE'),
        ]:
            for seq_type in (0, 1):
                mask = types == seq_type
                if not mask.any():
                    continue
                sc = ax.scatter(
                    z2d[mask, 0], z2d[mask, 1],
                    c=residues[mask], cmap='tab20',
                    marker=_TYPE_MARKER[seq_type],
                    s=22, alpha=0.75,
                    vmin=residues.min(), vmax=residues.max(),
                )
            ax.set_title(title)
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')

        # Shared colorbar on the right axis
        cbar = fig.colorbar(sc, ax=ax_al, label=f'a₀ + a₁·{p}')
        cbar.ax.tick_params(labelsize=8)

        # Type legend
        legend_handles = [
            Line2D([0], [0], marker=_TYPE_MARKER[t], color='grey',
                   markersize=7, linestyle='None', label=_TYPE_LABEL[t])
            for t in (0, 1)
        ]
        ax_al.legend(handles=legend_handles, loc='upper right')

        plot_path = os.path.join(save_dir, f'latent_space_p{p}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'saved → {plot_path}')


# ── Standalone entry point ────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description='Latent space visualization')
    p.add_argument('--unaligned_path', default='./checkpoints/beta_vae.pt')
    p.add_argument('--aligned_path',   default='./checkpoints/beta_vae_metric.pt')
    p.add_argument('--primes', nargs='+', type=int, default=[2, 3, 5, 7, 11])
    p.add_argument('--N',           type=int, default=64)
    p.add_argument('--vocab_size',  type=int, default=13)
    p.add_argument('--hidden_dim',  type=int, default=64)
    p.add_argument('--latent_dim',  type=int, default=32)
    p.add_argument('--num_samples', type=int, default=150)
    p.add_argument('--save_dir',    default='./plots')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    evaluate_and_plot_latents(
        unaligned_path=args.unaligned_path,
        aligned_path=args.aligned_path,
        save_dir=args.save_dir,
        N=args.N,
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        primes=args.primes,
        num_samples=args.num_samples,
    )
