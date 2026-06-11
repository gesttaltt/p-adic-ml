import argparse
import math
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from models import ConditionalVQVAE, PriorGRU
from hierarchical_vqvae import HierarchicalVQVAE, TopPriorGRU, BotPriorGRU
from hierarchical_3level import (ThreeLevelVQVAE, ThreeLevelTopPriorGRU,
                                 ThreeLevelMidPriorGRU, ThreeLevelBotPriorGRU)
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
    model_type='flat',
    hyperbolic_top=False,
    top_curvature=1.0,
    use_attention_decoder=False,
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
    if model_type == 'flat':
        vqvae = ConditionalVQVAE(
            vocab_size=vocab_size, hidden_dim=hidden_dim,
            codebook_size=codebook_size, latent_dim=latent_dim, N=N,
        )
        vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
        vqvae.to(device).eval()

        prior = PriorGRU(codebook_size=codebook_size, latent_dim=latent_dim, cond_dim=16)
        prior.load_state_dict(torch.load(prior_path, map_location=device))
        prior.to(device).eval()
    elif model_type == 'hierarchical':
        vqvae = HierarchicalVQVAE(
            vocab_size=vocab_size, hidden_dim=hidden_dim, N=N,
            bot_codebook=codebook_size, top_codebook=16,
            bot_dim=latent_dim, top_dim=latent_dim, cond_dim=16,
            hyperbolic_top=hyperbolic_top, top_curvature=top_curvature,
            use_attention_decoder=use_attention_decoder
        )
        vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
        vqvae.to(device).eval()

        if os.path.isdir(prior_path):
            top_p_path = os.path.join(prior_path, 'top_prior.pt')
            bot_p_path = os.path.join(prior_path, 'bot_prior.pt')
        else:
            dir_name = os.path.dirname(prior_path)
            top_p_path = os.path.join(dir_name, 'top_prior.pt')
            bot_p_path = os.path.join(dir_name, 'bot_prior.pt')

        top_prior = TopPriorGRU(top_codebook=16, top_dim=latent_dim, cond_dim=16, hidden_size=128, num_layers=2)
        top_prior.load_state_dict(torch.load(top_p_path, map_location=device))
        top_prior.to(device).eval()

        bot_prior = BotPriorGRU(bot_codebook=codebook_size, top_codebook=16, bot_dim=latent_dim, top_dim=latent_dim, cond_dim=16, hidden_size=256, num_layers=2)
        bot_prior.load_state_dict(torch.load(bot_p_path, map_location=device))
        bot_prior.to(device).eval()
    elif model_type == 'three_level':
        vqvae = ThreeLevelVQVAE(
            vocab_size=vocab_size, hidden_dim=hidden_dim, N=N,
            bot_codebook=codebook_size, mid_codebook=32, top_codebook=16,
            bot_dim=latent_dim, mid_dim=latent_dim, top_dim=latent_dim, cond_dim=16,
            use_attention_decoder=use_attention_decoder
        )
        vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
        vqvae.to(device).eval()

        if os.path.isdir(prior_path):
            top_p_path = os.path.join(prior_path, 'top_prior.pt')
            mid_p_path = os.path.join(prior_path, 'mid_prior.pt')
            bot_p_path = os.path.join(prior_path, 'bot_prior.pt')
        else:
            dir_name = os.path.dirname(prior_path)
            top_p_path = os.path.join(dir_name, 'top_prior.pt')
            mid_p_path = os.path.join(dir_name, 'mid_prior.pt')
            bot_p_path = os.path.join(dir_name, 'bot_prior.pt')

        top_prior = ThreeLevelTopPriorGRU(top_codebook=16, top_dim=latent_dim)
        top_prior.load_state_dict(torch.load(top_p_path, map_location=device))
        top_prior.to(device).eval()

        mid_prior = ThreeLevelMidPriorGRU(mid_codebook=32, top_codebook=16, mid_dim=latent_dim, top_dim=latent_dim)
        mid_prior.load_state_dict(torch.load(mid_p_path, map_location=device))
        mid_prior.to(device).eval()

        bot_prior = ThreeLevelBotPriorGRU(bot_codebook=codebook_size, mid_codebook=32, top_codebook=16, bot_dim=latent_dim, mid_dim=latent_dim, top_dim=latent_dim)
        bot_prior.load_state_dict(torch.load(bot_p_path, map_location=device))
        bot_prior.to(device).eval()

    # ── Real data ────────────────────────────────────────────────────────────
    ds = PadicDataset(primes=[p], N=N, num_samples_per_type=20)
    real_rats = [s['digits'].tolist() for s in ds if s['type'] == 0]
    real_algs = [s['digits'].tolist() for s in ds if s['type'] == 1]

    # ── Generate from prior ──────────────────────────────────────────────────
    p_tensor = torch.full((num_generate,), p, dtype=torch.long, device=device)
    with torch.no_grad():
        if model_type == 'flat':
            latent_indices = prior.sample(p_tensor, L=N // 2, temperature=0.7)
            quantized = vqvae.quantizer.embedding(latent_indices)
            logits = vqvae.decode(quantized, p_tensor)
        elif model_type == 'hierarchical':
            idx_top_s = top_prior.sample(p_tensor, L=N // 4, temperature=0.7)
            idx_bot_s = bot_prior.sample(idx_top_s, p_tensor, temperature=0.7)
            top_emb = vqvae.top_quantizer.embedding
            bot_emb = vqvae.bot_quantizer.embedding
            z_q_top = top_emb[idx_top_s] if not callable(top_emb) else top_emb(idx_top_s)
            z_q_bot = bot_emb[idx_bot_s] if not callable(bot_emb) else bot_emb(idx_bot_s)
            logits = vqvae.decode(z_q_bot, z_q_top, p_tensor)
        elif model_type == 'three_level':
            it = top_prior.sample(p_tensor, L=N // 8, temperature=0.7)
            im = mid_prior.sample(it, p_tensor, temperature=0.7)
            ib = bot_prior.sample(im, it, p_tensor, temperature=0.7)
            top_emb = vqvae.top_quantizer.embedding
            mid_emb = vqvae.mid_quantizer.embedding
            bot_emb = vqvae.bot_quantizer.embedding
            z_top = top_emb[it] if not callable(top_emb) else top_emb(it)
            z_mid = mid_emb[im] if not callable(mid_emb) else mid_emb(im)
            z_bot = bot_emb[ib] if not callable(bot_emb) else bot_emb(ib)
            logits = vqvae.decode(z_bot, z_mid, z_top, p_tensor)

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
            if model_type == 'flat':
                recon_logits, _, _ = vqvae(recon_data, p_eval)
            elif model_type == 'hierarchical':
                recon_logits, _, _, _ = vqvae(recon_data, p_eval)
            elif model_type == 'three_level':
                recon_logits, _, _, _, _ = vqvae(recon_data, p_eval)
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
    p.add_argument('--model_type',    choices=['flat', 'hierarchical', 'three_level'], default='flat')
    p.add_argument('--hyperbolic_top', action='store_true')
    p.add_argument('--top_curvature',  type=float, default=1.0)
    p.add_argument('--attention_decoder', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # vocab_size dynamic check:
    vocab_sz = max(max(args.primes) + 2, args.vocab_size)
    
    plot_padic_trees(
        vqvae_path=args.vqvae_path,
        prior_path=args.prior_path,
        primes=args.primes,
        save_dir=args.save_dir,
        N=args.N,
        device=device,
        vocab_size=vocab_sz,
        hidden_dim=args.hidden_dim,
        codebook_size=args.codebook_size,
        latent_dim=args.latent_dim,
        max_depth=args.max_depth,
        num_generate=args.num_generate,
        model_type=args.model_type,
        hyperbolic_top=args.hyperbolic_top,
        top_curvature=args.top_curvature,
        use_attention_decoder=args.attention_decoder,
    )
