"""
eval_conditional_alignment.py  —  Item #20

Conditional metric alignment for the hierarchical VQ-VAE.

The global alignment test (item #18) measured whether pairwise Euclidean
distances in the flattened bottom-code space correlate with p-adic distances
across ALL sequence pairs.  That test gave Spearman r ≈ 0.05 — poor, because
the bottom codes are conditional representations (within-bucket variation only).

This script measures the *correct* quantities:

  1. TOP-CODE BRANCH ALIGNMENT
     Within each top-code bucket, are sequences p-adically closer to each
     other than random cross-bucket pairs?
     → confirms top codes ≈ branch selectors at the metric level

  2. CONDITIONAL BOTTOM-CODE ALIGNMENT
     Within sequences that share the same top code AND the same prime, does
     Euclidean bottom-code distance correlate with p-adic distance?
     → tests whether bottom codes locally organise p-adic distances

Reports per-prime conditional Spearman r, unconditional r (baseline), and
the top-code branch separation ratio (within-bucket / cross-bucket p-adic dist).

No training required — loads ./checkpoints/hierarchical/vqvae.pt.

Output:  plots/conditional_alignment.md
"""

import os, math, torch
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

from dataset import PadicDataset
from hierarchical_vqvae import HierarchicalVQVAE
from metric_alignment import batch_padic_distance, compute_metric_loss

# ── config ────────────────────────────────────────────────────────────────────
CKPT        = './checkpoints/hierarchical/vqvae.pt'
PRIMES      = [2, 3, 5, 7, 11]
VOCAB       = max(PRIMES) + 2
N           = 64
SAMPLES     = 300          # per type per prime
MIN_BUCKET  = 10           # min sequences per (code, prime) bucket to include
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── load ──────────────────────────────────────────────────────────────────────

def load_model():
    m = HierarchicalVQVAE(vocab_size=VOCAB, hidden_dim=64, N=N)
    m.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    return m.to(DEVICE).eval()


# ── encode all sequences ──────────────────────────────────────────────────────

@torch.no_grad()
def encode_all(model, primes, samples_per_type):
    records = []
    for p_val in primes:
        ds = PadicDataset(primes=[p_val], N=N, num_samples_per_type=samples_per_type)
        for start in range(0, len(ds), 256):
            end   = min(start + 256, len(ds))
            batch = ds[start:end]
            digs  = torch.stack([b['digits'] for b in batch]).to(DEVICE)
            p_t   = torch.full((len(batch),), p_val, dtype=torch.long, device=DEVICE)

            z_q_bot, _, _, idx_top, _, _ = model.encode(digs, p_t)

            for i in range(len(batch)):
                from collections import Counter
                top_code = Counter(idx_top[i].cpu().tolist()).most_common(1)[0][0]
                records.append({
                    'prime':    p_val,
                    'seq_type': batch[i]['type'],
                    'top_code': top_code,
                    'digits':   digs[i].cpu(),
                    'z_bot':    z_q_bot[i].reshape(-1).cpu(),  # flatten [L_bot, D]
                })
    return records


# ── metric helpers ────────────────────────────────────────────────────────────

def spearman_z_vs_padic(z_list, digit_list, p_val):
    """Spearman r between pairwise Euclidean z-distance and p-adic distance."""
    if len(z_list) < 5:
        return float('nan')
    z      = torch.stack(z_list)                                    # [B, D]
    digits = torch.stack(digit_list)                                # [B, N]
    p_t    = torch.full((len(z_list),), p_val, dtype=torch.long)

    d_lat = (z.unsqueeze(1) - z.unsqueeze(0)).norm(dim=-1).numpy()
    d_pad = batch_padic_distance(digits, p_t).numpy()
    idx   = np.triu_indices(len(z_list), k=1)
    if d_pad[idx].std() < 1e-9 or d_lat[idx].std() < 1e-9:
        return float('nan')
    r, _  = spearmanr(d_pad[idx], d_lat[idx])
    return float(r)


def mean_padic(digit_list, p_val):
    """Mean pairwise p-adic distance."""
    if len(digit_list) < 2:
        return float('nan')
    digits = torch.stack(digit_list)
    p_t    = torch.full((len(digit_list),), p_val, dtype=torch.long)
    D      = batch_padic_distance(digits, p_t).numpy()
    idx    = np.triu_indices(len(digit_list), k=1)
    return float(D[idx].mean())


# ── main analysis ─────────────────────────────────────────────────────────────

def main():
    print(f'Device: {DEVICE}')
    model = load_model()

    print(f'\nEncoding {len(PRIMES)} × 3 × {SAMPLES} sequences...')
    records = encode_all(model, PRIMES, SAMPLES)
    print(f'Total: {len(records)} sequences')

    # ── 1. Top-code branch alignment ──────────────────────────────────────────
    print('\n── Top-code branch alignment ──────────────────────────────')

    branch_results = {}   # prime → {within_mean, cross_mean, ratio}
    for p_val in PRIMES:
        recs_p = [r for r in records if r['prime'] == p_val]

        # within-bucket distance: pairs sharing the same top code
        by_code = defaultdict(list)
        for r in recs_p:
            by_code[r['top_code']].append(r['digits'])

        within_d = []
        for code, digs in by_code.items():
            if len(digs) >= 2:
                within_d.append(mean_padic(digs, p_val))

        # cross-bucket distance: random pairs from different codes
        rng = np.random.default_rng(42)
        cross_d = []
        for _ in range(300):
            i, j = rng.choice(len(recs_p), 2, replace=False)
            if recs_p[i]['top_code'] != recs_p[j]['top_code']:
                cross_d.append(
                    batch_padic_distance(
                        torch.stack([recs_p[i]['digits'], recs_p[j]['digits']]),
                        torch.tensor([p_val, p_val])
                    )[0, 1].item()
                )

        w = float(np.nanmean(within_d)) if within_d else float('nan')
        c = float(np.mean(cross_d))     if cross_d  else float('nan')
        ratio = w / c if c > 0 else float('nan')
        branch_results[p_val] = {'within': w, 'cross': c, 'ratio': ratio}
        print(f'  p={p_val}: within={w:.4f}  cross={c:.4f}  ratio={ratio:.3f}')

    # ── 2. Conditional bottom-code alignment ──────────────────────────────────
    print('\n── Conditional bottom-code alignment (Spearman r) ─────────')

    # Unconditional baseline per prime
    uncond_r = {}
    for p_val in PRIMES:
        rp = [r for r in records if r['prime'] == p_val]
        uncond_r[p_val] = spearman_z_vs_padic(
            [r['z_bot'] for r in rp], [r['digits'] for r in rp], p_val
        )
    print(f'  Unconditional r: ' +
          ' '.join(f'p={p} {uncond_r[p]:.4f}' for p in PRIMES))

    # Conditional r: within (top_code, prime) buckets
    cond_r_per_prime = defaultdict(list)
    cond_n_per_prime = defaultdict(int)

    by_code_prime = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_code_prime[r['top_code']][r['prime']].append(r)

    for code in range(16):
        for p_val in PRIMES:
            bucket = by_code_prime[code][p_val]
            if len(bucket) < MIN_BUCKET:
                continue
            r_val = spearman_z_vs_padic(
                [b['z_bot'] for b in bucket],
                [b['digits'] for b in bucket],
                p_val,
            )
            if not math.isnan(r_val):
                cond_r_per_prime[p_val].append(r_val)
                cond_n_per_prime[p_val] += len(bucket)

    print(f'\n  {"Prime":<8} {"Uncond r":>10} {"Cond r (mean)":>15} {"n buckets":>12} {"n seqs":>10}')
    print(f'  {"-"*58}')
    for p_val in PRIMES:
        crs = cond_r_per_prime[p_val]
        mean_r = float(np.mean(crs)) if crs else float('nan')
        print(f'  p={p_val:<6} {uncond_r[p_val]:>10.4f} {mean_r:>15.4f} '
              f'{len(crs):>12} {cond_n_per_prime[p_val]:>10}')

    # ── save report ───────────────────────────────────────────────────────────
    lines = ['# Conditional Metric Alignment — Hierarchical VQ-VAE\n',
             f'Checkpoint: `{CKPT}`  N={N}  {SAMPLES} samples/type/prime\n',
             '## 1. Top-Code Branch Alignment\n',
             '| Prime | Within-bucket dist | Cross-bucket dist | Ratio (within/cross) |',
             '| :---: | :---: | :---: | :---: |']
    for p_val in PRIMES:
        b = branch_results[p_val]
        flag = '✓ tight' if b['ratio'] < 1.0 else '✗ loose'
        lines.append(f'| $p={p_val}$ | {b["within"]:.4f} | {b["cross"]:.4f} '
                     f'| {b["ratio"]:.3f} ({flag}) |')

    lines += ['\n## 2. Conditional Bottom-Code Alignment\n',
              '| Prime | Unconditional $r$ | Conditional $r$ (mean) | Gain | Buckets used |',
              '| :---: | :---: | :---: | :---: | :---: |']
    for p_val in PRIMES:
        crs   = cond_r_per_prime[p_val]
        mean_r = float(np.mean(crs)) if crs else float('nan')
        gain   = mean_r - uncond_r[p_val] if not math.isnan(mean_r) else float('nan')
        flag   = '↑' if gain > 0.02 else ('≈' if abs(gain) <= 0.02 else '↓')
        lines.append(f'| $p={p_val}$ | {uncond_r[p_val]:.4f} | {mean_r:.4f} '
                     f'| {gain:+.4f} {flag} | {len(crs)} |')

    content = '\n'.join(lines)
    os.makedirs('./plots', exist_ok=True)
    with open('./plots/conditional_alignment.md', 'w') as f:
        f.write(content + '\n')
    print(f'\nSaved to ./plots/conditional_alignment.md')
    print('\n' + content)


if __name__ == '__main__':
    main()
