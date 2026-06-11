import argparse
import math
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from models import ConditionalVQVAE, PriorGRU
from dataset import PadicDataset
from padic_math import padic_to_float
from viz_style import (apply_style, RATIONAL_COLOR, ALGEBRAIC_COLOR,
                       GENERATED_COLOR, seq_type_legend_handles)

apply_style()


def get_path_coords(digits, p, max_depth=8, r=0.75, initial_spread=math.pi * 0.6):
    """
    Compute 2D Cartesian coordinates for a p-adic digit sequence as a path
    down a p-ary tree.  Returns (x_list, y_list) of length min(len(digits),
    max_depth) + 1 (includes the root at (0, 0)).
    """
    digits = digits[:max_depth]
    x, y = [0.0], [0.0]
    curr_x, curr_y = 0.0, 0.0
    curr_theta = math.pi / 2
    spread = initial_spread

    for i, d in enumerate(digits):
        ratio = (d - (p - 1) / 2.0) / (p / 2.0) if p > 1 else 0.0
        theta = curr_theta + ratio * (spread / 2.0)
        length = r ** i
        next_x = curr_x + length * math.cos(theta)
        next_y = curr_y + length * math.sin(theta)
        x.append(next_x)
        y.append(next_y)
        curr_x, curr_y = next_x, next_y
        curr_theta = theta
        spread *= 0.75

    return x, y


def check_periodicity(digits):
    """
    Return (period_len, start_idx) if the sequence is short-period periodic,
    else (None, None).
    """
    n = len(digits)
    for p_len in range(1, n // 2 + 1):
        for start in range(n - 2 * p_len):
            pattern = digits[start:start + p_len]
            is_periodic = True
            for i in range(start, n - p_len, p_len):
                chunk = digits[i:i + p_len]
                compare_len = min(p_len, len(chunk))
                if chunk[:compare_len] != pattern[:compare_len]:
                    is_periodic = False
                    break
            if is_periodic and pattern:
                return p_len, start
    return None, None


def plot_padic_tree(
    vqvae_path,
    prior_path,
    p,
    save_path=None,
    save_dir='./plots',
    N=64,
    device='cpu',
    vocab_size=13,
    hidden_dim=64,
    codebook_size=64,
    latent_dim=32,
    max_depth=8,
    num_generate=50,
    verbose=True,
):
    """
    Generate and save a single p-adic tree visualization.

    Parameters
    ----------
    save_path : str or None
        Explicit output path.  If None, auto-generates
        ``{save_dir}/padic_tree_{p}.png``.
    """
    if save_path is None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'padic_tree_{p}.png')
    else:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────────
    vqvae = ConditionalVQVAE(
        vocab_size=vocab_size, hidden_dim=hidden_dim,
        codebook_size=codebook_size, latent_dim=latent_dim, N=N,
    )
    vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
    vqvae.to(device).eval()

    prior = PriorGRU(codebook_size=codebook_size, latent_dim=latent_dim, cond_dim=16)
    prior.load_state_dict(torch.load(prior_path, map_location=device))
    prior.to(device).eval()

    # ── Real data ────────────────────────────────────────────────────────────
    ds = PadicDataset(primes=[p], N=N, num_samples_per_type=20)
    real_rats = [s['digits'].tolist() for s in ds if s['type'] == 0]
    real_algs = [s['digits'].tolist() for s in ds if s['type'] == 1]

    # ── Generate from prior ──────────────────────────────────────────────────
    p_tensor = torch.full((num_generate,), p, dtype=torch.long, device=device)
    with torch.no_grad():
        latent_indices = prior.sample(p_tensor, L=N // 2, temperature=0.7)
        quantized = vqvae.quantizer.embedding(latent_indices)
        logits = vqvae.decode(quantized, p_tensor)
        generated_digits = torch.argmax(logits, dim=-1).cpu().numpy()

    if verbose:
        valid = sum(1 for seq in generated_digits if all(d < p for d in seq.tolist()))
        periodic = sum(
            1 for seq in generated_digits
            if check_periodicity(seq.tolist())[0] is not None
            and check_periodicity(seq.tolist())[0] < 10
        )
        recon_data = torch.tensor(real_rats + real_algs, dtype=torch.long, device=device)
        p_eval = torch.full((recon_data.shape[0],), p, dtype=torch.long, device=device)
        with torch.no_grad():
            recon_logits, _, _ = vqvae(recon_data, p_eval)
            acc = (torch.argmax(recon_logits, dim=-1) == recon_data).float().mean().item()
        print(f"p={p}  recon_acc={acc*100:.1f}%  "
              f"valid={valid}/{num_generate}  "
              f"periodic={periodic}/{num_generate}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'{p}-adic Tree — depth {max_depth}', pad=10)

    for seq in real_rats[:15]:
        x, y = get_path_coords(seq, p, max_depth=max_depth)
        ax.plot(x, y, color=RATIONAL_COLOR, alpha=0.55, linewidth=1.4, zorder=2)

    for seq in real_algs[:15]:
        x, y = get_path_coords(seq, p, max_depth=max_depth)
        ax.plot(x, y, color=ALGEBRAIC_COLOR, alpha=0.55, linewidth=1.4, zorder=2)

    for seq in generated_digits[:30]:
        x, y = get_path_coords(seq.tolist(), p, max_depth=max_depth)
        ax.plot(x, y, color=GENERATED_COLOR, alpha=0.60, linewidth=1.0, zorder=3)
        ax.scatter(x[-1], y[-1], color=GENERATED_COLOR, s=12, alpha=0.85, zorder=4)

    ax.legend(handles=seq_type_legend_handles(), loc='lower right',
              framealpha=0.9, fontsize=9)

    plt.savefig(save_path)
    plt.close()
    if verbose:
        print(f'Saved → {save_path}')


def plot_padic_trees(
    vqvae_path,
    prior_path,
    primes,
    save_dir='./plots',
    **kwargs,
):
    """Call plot_padic_tree for each prime in *primes*."""
    for p in primes:
        plot_padic_tree(vqvae_path, prior_path, p, save_dir=save_dir, **kwargs)


# ── Standalone entry point ────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description='p-adic tree visualization')
    p.add_argument('--vqvae_path',    default='./checkpoints/vqvae.pt')
    p.add_argument('--prior_path',    default='./checkpoints/prior.pt')
    p.add_argument('--primes',        nargs='+', type=int, default=[2, 3, 5, 7, 11])
    p.add_argument('--N',             type=int, default=64)
    p.add_argument('--vocab_size',    type=int, default=13)
    p.add_argument('--hidden_dim',    type=int, default=64)
    p.add_argument('--codebook_size', type=int, default=64)
    p.add_argument('--latent_dim',    type=int, default=32)
    p.add_argument('--max_depth',     type=int, default=8)
    p.add_argument('--num_generate',  type=int, default=50)
    p.add_argument('--save_dir',      default='./plots')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plot_padic_trees(
        vqvae_path=args.vqvae_path,
        prior_path=args.prior_path,
        primes=args.primes,
        save_dir=args.save_dir,
        N=args.N,
        device=device,
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        codebook_size=args.codebook_size,
        latent_dim=args.latent_dim,
        max_depth=args.max_depth,
        num_generate=args.num_generate,
    )
