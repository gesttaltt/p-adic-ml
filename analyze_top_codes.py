"""
analyze_top_codes.py

Interpretability analysis of the 16 top-codebook entries in HierarchicalVQVAE.

For each top code k ∈ {0..15} we collect all sequences assigned to it and report:

  1. Prime distribution      — which primes dominate this code?
  2. Sequence-type breakdown — rational / algebraic / random share
  3. First-digit prefix      — most common digit[0] and digit[0:2] prefix
  4. Distance structure      — mean within-code p-adic distance vs cross-code

Then we run the conditional coherence test (item #15):
  5. Fix each top code, sample 50 bottom sequences, measure within-code
     p-adic distance and compare to random (unconditional) baseline.

Outputs:
  plots/top_code_analysis.png  — heatmaps + distance bar charts
  plots/top_code_analysis.md   — full markdown report
"""

import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from dataset import PadicDataset
from hierarchical_vqvae import HierarchicalVQVAE, TopPriorGRU, BotPriorGRU
from metric_alignment import batch_padic_distance

# ── config ────────────────────────────────────────────────────────────────────
CKPT_DIR      = './checkpoints/hierarchical'
PRIMES        = [2, 3, 5, 7, 11]
N             = 64
SAMPLES_PER   = 300          # per type per prime for the analysis dataset
TOP_CODEBOOK  = 16
BOT_CODEBOOK  = 64
COND_SAMPLES  = 50           # samples per top code in coherence test
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── helpers ───────────────────────────────────────────────────────────────────

def load_models():
    vocab = max(PRIMES) + 2
    vqvae = HierarchicalVQVAE(vocab_size=vocab, hidden_dim=64, N=N,
                               bot_codebook=BOT_CODEBOOK, top_codebook=TOP_CODEBOOK)
    vqvae.load_state_dict(torch.load(f'{CKPT_DIR}/vqvae.pt', map_location=DEVICE))
    vqvae.to(DEVICE).eval()

    top_prior = TopPriorGRU(top_codebook=TOP_CODEBOOK, top_dim=32)
    top_prior.load_state_dict(torch.load(f'{CKPT_DIR}/top_prior.pt', map_location=DEVICE))
    top_prior.to(DEVICE).eval()

    bot_prior = BotPriorGRU(bot_codebook=BOT_CODEBOOK, top_codebook=TOP_CODEBOOK,
                             bot_dim=32, top_dim=32)
    bot_prior.load_state_dict(torch.load(f'{CKPT_DIR}/bot_prior.pt', map_location=DEVICE))
    bot_prior.to(DEVICE).eval()

    return vqvae, top_prior, bot_prior


def padic_dist_scalar(a, b, p):
    """p-adic distance between two sequences."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return float(p ** -i)
    return 0.0


# ── Step 1-4: assign every dataset sequence to a top code ─────────────────────

@torch.no_grad()
def assign_top_codes(vqvae, primes, n_samples):
    """
    Returns a list of dicts, one per sequence:
      { 'top_code': int, 'prime': int, 'seq_type': int, 'digits': list }
    """
    records = []
    for p_val in primes:
        ds = PadicDataset(primes=[p_val], N=N, num_samples_per_type=n_samples)
        digits_t = torch.stack([s['digits'] for s in ds]).to(DEVICE)
        p_t      = torch.full((len(ds),), p_val, dtype=torch.long, device=DEVICE)
        types    = [s['type'] for s in ds]

        # encode → get top indices
        # process in batches of 256
        for start in range(0, len(ds), 256):
            end = min(start + 256, len(ds))
            d_b = digits_t[start:end]
            p_b = p_t[start:end]
            _, _, _, idx_top, _, _ = vqvae.encode(d_b, p_b)
            # idx_top: [B, L_top=16]; use the *mean mode* top code per sequence
            # (each sequence maps to L_top top tokens — summarise by majority vote)
            for i, (row_top, seq_type) in enumerate(zip(idx_top.cpu().numpy(),
                                                        types[start:end])):
                code = Counter(row_top.tolist()).most_common(1)[0][0]
                records.append({
                    'top_code':  code,
                    'prime':     p_val,
                    'seq_type':  seq_type,
                    'digits':    digits_t[start + i].cpu().numpy().tolist(),
                })
    return records


# ── Step 4: distance structure ─────────────────────────────────────────────────

def compute_distance_structure(records, primes):
    """
    For each top code: mean within-code p-adic distance.
    Also compute cross-code baseline (random pairs).
    Only compare pairs with the SAME prime (p-adic distance is only defined within a base).
    """
    by_code_and_prime = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_code_and_prime[r['top_code']][r['prime']].append(r['digits'])

    within_dists = {}   # code -> mean p-adic dist
    n_pairs_used = {}

    for code in range(TOP_CODEBOOK):
        all_d = []
        for p_val in primes:
            seqs = by_code_and_prime[code][p_val]
            if len(seqs) < 2:
                continue
            # sample up to 30 pairs
            rng = np.random.default_rng(42)
            idx = rng.choice(len(seqs), size=min(30, len(seqs)), replace=False)
            chosen = [seqs[i] for i in idx]
            for i in range(len(chosen)):
                for j in range(i + 1, len(chosen)):
                    all_d.append(padic_dist_scalar(chosen[i], chosen[j], p_val))
        within_dists[code]  = float(np.mean(all_d)) if all_d else float('nan')
        n_pairs_used[code]  = len(all_d)

    # Cross-code baseline: random pairs from different codes, same prime
    cross_d = []
    rng = np.random.default_rng(0)
    for p_val in primes:
        all_seqs = [(r['digits'], r['top_code']) for r in records if r['prime'] == p_val]
        if len(all_seqs) < 2:
            continue
        for _ in range(200):
            i, j = rng.choice(len(all_seqs), 2, replace=False)
            if all_seqs[i][1] != all_seqs[j][1]:
                cross_d.append(padic_dist_scalar(all_seqs[i][0], all_seqs[j][0], p_val))
    cross_mean = float(np.mean(cross_d)) if cross_d else float('nan')

    return within_dists, cross_mean, n_pairs_used


# ── Step 5: conditional coherence test ────────────────────────────────────────

@torch.no_grad()
def conditional_coherence(vqvae, bot_prior, p_val=5):
    """
    For each of the 16 top codes, fix that code for all L_top positions,
    sample COND_SAMPLES bottom sequences, decode, compute mean pairwise p-adic dist.
    Compare to unconditional baseline (random top codes).
    """
    p_t   = torch.full((COND_SAMPLES,), p_val, dtype=torch.long, device=DEVICE)
    L_top = N // 4
    L_bot = N // 2

    results = {}
    for code in range(TOP_CODEBOOK):
        idx_top = torch.full((COND_SAMPLES, L_top), code, dtype=torch.long, device=DEVICE)
        idx_bot = bot_prior.sample(idx_top, p_t, temperature=0.8)

        # decode to digit sequences
        z_q_top = vqvae.top_quantizer.embedding(idx_top)
        z_q_bot = vqvae.bot_quantizer.embedding(idx_bot)
        logits  = vqvae.decode(z_q_bot, z_q_top, p_t)
        seqs    = torch.argmax(logits, dim=-1)  # [B, N]

        # pairwise p-adic distance
        p_ten = p_t
        D     = batch_padic_distance(seqs, p_ten).cpu().numpy()
        idx   = np.triu_indices(COND_SAMPLES, k=1)
        results[code] = float(np.mean(D[idx]))

    # unconditional baseline: random top codes, then sample
    rng     = torch.Generator(device=DEVICE)
    idx_top_rand = torch.randint(TOP_CODEBOOK, (COND_SAMPLES, L_top),
                                 generator=rng, device=DEVICE)
    idx_bot_rand = bot_prior.sample(idx_top_rand, p_t, temperature=0.8)
    z_q_top_r = vqvae.top_quantizer.embedding(idx_top_rand)
    z_q_bot_r = vqvae.bot_quantizer.embedding(idx_bot_rand)
    logits_r  = vqvae.decode(z_q_bot_r, z_q_top_r, p_t)
    seqs_r    = torch.argmax(logits_r, dim=-1)
    D_r       = batch_padic_distance(seqs_r, p_t).cpu().numpy()
    baseline  = float(np.mean(D_r[np.triu_indices(COND_SAMPLES, k=1)]))

    return results, baseline


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_plots(records, within_dists, cross_mean, coh_results, coh_baseline, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=130)

    # 1. Prime heatmap: rows=top codes, cols=primes
    prime_mat = np.zeros((TOP_CODEBOOK, len(PRIMES)))
    type_mat  = np.zeros((TOP_CODEBOOK, 3))
    for r in records:
        c = r['top_code']
        prime_mat[c, PRIMES.index(r['prime'])] += 1
        type_mat[c, r['seq_type']] += 1

    # normalize rows
    row_sums = prime_mat.sum(axis=1, keepdims=True) + 1e-9
    prime_mat_n = prime_mat / row_sums
    type_row = type_mat / (type_mat.sum(axis=1, keepdims=True) + 1e-9)

    ax = axes[0, 0]
    im = ax.imshow(prime_mat_n, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(PRIMES))); ax.set_xticklabels([f'p={p}' for p in PRIMES])
    ax.set_yticks(range(TOP_CODEBOOK)); ax.set_yticklabels([str(c) for c in range(TOP_CODEBOOK)], fontsize=7)
    ax.set_title('Prime distribution per top code\n(fraction of sequences)', fontweight='bold')
    ax.set_xlabel('Prime'); ax.set_ylabel('Top code')
    fig.colorbar(im, ax=ax)

    # 2. Sequence-type heatmap
    ax = axes[0, 1]
    im2 = ax.imshow(type_row, aspect='auto', cmap='Oranges', vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['Rational', 'Algebraic', 'Random'])
    ax.set_yticks(range(TOP_CODEBOOK)); ax.set_yticklabels([str(c) for c in range(TOP_CODEBOOK)], fontsize=7)
    ax.set_title('Sequence type per top code\n(fraction)', fontweight='bold')
    ax.set_xlabel('Type'); ax.set_ylabel('Top code')
    fig.colorbar(im2, ax=ax)

    # 3. Within-code vs cross-code distance
    ax = axes[1, 0]
    codes  = list(range(TOP_CODEBOOK))
    wd     = [within_dists.get(c, float('nan')) for c in codes]
    colors = ['#e57373' if w > cross_mean else '#81c784' for w in wd]
    bars   = ax.bar(codes, wd, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(cross_mean, color='navy', linestyle='--', linewidth=1.5,
               label=f'Cross-code baseline ({cross_mean:.4f})')
    ax.set_xlabel('Top code'); ax.set_ylabel('Mean p-adic distance')
    ax.set_title('Within-code p-adic distance\n(green = below cross-code baseline)', fontweight='bold')
    ax.set_xticks(codes); ax.set_xticklabels([str(c) for c in codes], fontsize=8)
    ax.legend(fontsize=8)

    # 4. Conditional coherence
    ax = axes[1, 1]
    coh_vals = [coh_results.get(c, float('nan')) for c in codes]
    colors2  = ['#81c784' if v < coh_baseline else '#e57373' for v in coh_vals]
    ax.bar(codes, coh_vals, color=colors2, edgecolor='black', linewidth=0.5)
    ax.axhline(coh_baseline, color='navy', linestyle='--', linewidth=1.5,
               label=f'Unconditional baseline ({coh_baseline:.4f})')
    ax.set_xlabel('Top code'); ax.set_ylabel('Mean pairwise p-adic distance')
    ax.set_title('Conditional coherence (p=5)\n(green = tighter than unconditional)', fontweight='bold')
    ax.set_xticks(codes); ax.set_xticklabels([str(c) for c in codes], fontsize=8)
    ax.legend(fontsize=8)

    plt.suptitle('Top Codebook Interpretability Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to {save_path}')


# ── Markdown report ───────────────────────────────────────────────────────────

def make_report(records, within_dists, cross_mean, coh_results, coh_baseline, save_path):
    by_code = defaultdict(list)
    for r in records:
        by_code[r['top_code']].append(r)

    lines = ['# Top Codebook Interpretability Analysis\n',
             f'Model: `{CKPT_DIR}`  ',
             f'Dataset: Broad-11, N={N}, {SAMPLES_PER} samples/type/prime\n',
             f'Cross-code distance baseline: **{cross_mean:.4f}**  ',
             f'Unconditional coherence baseline (p=5): **{coh_baseline:.4f}**\n',
             '| Code | N | Top prime | Top type | Common prefix | '
             'Within dist | vs baseline | Coh dist | Coh vs baseline |',
             '| :---: | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |']

    n_tight  = 0
    n_coh    = 0

    for code in range(TOP_CODEBOOK):
        recs = by_code[code]
        if not recs:
            lines.append(f'| {code} | 0 | — | — | — | — | — | — | — |')
            continue

        prime_ctr = Counter(r['prime']    for r in recs)
        type_ctr  = Counter(r['seq_type'] for r in recs)
        prefix_ctr = Counter(tuple(r['digits'][:2]) for r in recs)

        top_prime  = prime_ctr.most_common(1)[0][0]
        top_type   = ['Rational', 'Algebraic', 'Random'][type_ctr.most_common(1)[0][0]]
        top_prefix = list(prefix_ctr.most_common(1)[0][0])
        top_pct    = prime_ctr.most_common(1)[0][1] / len(recs)

        wd   = within_dists.get(code, float('nan'))
        cd   = coh_results.get(code, float('nan'))

        wd_flag  = '✓ tight'  if (not math.isnan(wd)  and wd  < cross_mean)   else '✗ loose'
        cd_flag  = '✓ tight'  if (not math.isnan(cd)  and cd  < coh_baseline) else '✗ loose'

        if wd_flag == '✓ tight': n_tight += 1
        if cd_flag == '✓ tight': n_coh   += 1

        lines.append(
            f'| {code} | {len(recs)} | p={top_prime} ({top_pct:.0%}) | {top_type} | '
            f'{top_prefix} | {wd:.4f} | {wd_flag} | {cd:.4f} | {cd_flag} |'
        )

    lines += [
        f'\n**Within-code distance tighter than cross-code baseline**: {n_tight}/{TOP_CODEBOOK} codes',
        f'**Conditional coherence tighter than unconditional baseline**: {n_coh}/{TOP_CODEBOOK} codes',
    ]

    content = '\n'.join(lines)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(content + '\n')
    print(f'Saved report to {save_path}')
    return content, n_tight, n_coh


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f'Device: {DEVICE}')
    vqvae, top_prior, bot_prior = load_models()

    print(f'\nAssigning {len(PRIMES)} × 3 × {SAMPLES_PER} sequences to top codes...')
    records = assign_top_codes(vqvae, PRIMES, SAMPLES_PER)
    print(f'Total sequences: {len(records)}')

    code_counts = Counter(r['top_code'] for r in records)
    print('\nTop-code assignment counts:')
    for c in range(TOP_CODEBOOK):
        bar = '█' * (code_counts[c] // 20)
        print(f'  code {c:2d}: {code_counts[c]:5d}  {bar}')

    print('\nComputing distance structure...')
    within_dists, cross_mean, n_pairs = compute_distance_structure(records, PRIMES)
    print(f'Cross-code baseline: {cross_mean:.4f}')
    n_tight_dist = sum(1 for c, d in within_dists.items()
                       if not math.isnan(d) and d < cross_mean)
    print(f'Codes with within < cross: {n_tight_dist}/{TOP_CODEBOOK}')

    print('\nRunning conditional coherence test (p=5)...')
    coh_results, coh_baseline = conditional_coherence(vqvae, bot_prior, p_val=5)
    print(f'Unconditional baseline: {coh_baseline:.4f}')
    n_coh = sum(1 for d in coh_results.values() if d < coh_baseline)
    print(f'Codes tighter than baseline: {n_coh}/{TOP_CODEBOOK}')

    make_plots(records, within_dists, cross_mean, coh_results, coh_baseline,
               './plots/top_code_analysis.png')

    content, n_tight, n_coh2 = make_report(
        records, within_dists, cross_mean, coh_results, coh_baseline,
        './plots/top_code_analysis.md')

    print('\n' + content)


if __name__ == '__main__':
    main()
