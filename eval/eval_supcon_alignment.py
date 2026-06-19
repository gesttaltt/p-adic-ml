"""
eval_supcon_alignment.py

Evaluates the metric_proj head of a ThreeLevelVQVAE checkpoint on three axes:
  1. Spearman r     — rank correlation between latent distance and p-adic distance
  2. kNN accuracy   — can k-nearest-neighbor predict the prime from z_proj?
  3. Silhouette     — how well-separated are the prime clusters in z_proj space?

Usage:
  python eval/eval_supcon_alignment.py --checkpoint ./checkpoints/hierarchical_3level_supcon
  python eval/eval_supcon_alignment.py \\
      --checkpoint ./checkpoints/hierarchical_3level_warmup40 \\
      --label "MSE-40ep"
"""
import sys, os
_r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_r, 'src'))
os.chdir(_r)

import argparse
import torch
import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from dataset import PadicDataset
from hierarchical_3level import ThreeLevelVQVAE
from metric_alignment import batch_padic_distance

PRIMES  = [2, 3, 5, 7, 11]
N       = 128
VOCAB   = max(PRIMES) + 2
SAMPLES = 200   # per type per prime (type != random)


def load_model(checkpoint_dir):
    model = ThreeLevelVQVAE(
        vocab_size=VOCAB, hidden_dim=64, N=N,
        bot_codebook=64, mid_codebook=32, top_codebook=16,
        use_attention_decoder=True,
    )
    path = os.path.join(checkpoint_dir, 'vqvae.pt')
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model.eval()


@torch.no_grad()
def extract_representations(model, primes, samples_per_type):
    all_z_raw  = []
    all_z_proj = []
    all_digits = []
    all_p      = []

    for p_val in primes:
        ds   = PadicDataset(primes=[p_val], N=N, num_samples_per_type=samples_per_type)
        seqs = [s for s in ds if s['type'] != 2]   # exclude random

        digits = torch.stack([s['digits'] for s in seqs])
        p_t    = torch.full((len(seqs),), p_val, dtype=torch.long)

        z_q_bot, _, _, _, _, _, _ = model.encode(digits, p_t)
        z_pooled = z_q_bot.mean(dim=1)              # [B, bot_dim]
        z_proj   = model.metric_proj(z_pooled)      # [B, bot_dim]

        all_z_raw.append(z_pooled)
        all_z_proj.append(z_proj)
        all_digits.append(digits)
        all_p.append(p_t)

    return (torch.cat(all_z_raw), torch.cat(all_z_proj),
            torch.cat(all_digits), torch.cat(all_p))


def spearman_per_prime(z, digits, p_tensor, primes):
    results = {}
    for p_val in primes:
        mask = p_tensor == p_val
        if mask.sum() < 4:
            continue
        z_p   = z[mask]
        dig_p = digits[mask]
        p_p   = p_tensor[mask]

        d_pad = batch_padic_distance(dig_p, p_p).numpy()
        d_lat = (z_p.unsqueeze(1) - z_p.unsqueeze(0)).norm(dim=-1).numpy()
        idx   = np.triu_indices(mask.sum(), k=1)
        r, _  = spearmanr(d_pad[idx], d_lat[idx])
        results[p_val] = float(r)
    return results


def knn_prime_accuracy(z_np, labels, k=5):
    clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    # leave-one-out cross-val approximation via train=test (optimistic but comparable)
    clf.fit(z_np, labels)
    return clf.score(z_np, labels)


def centroid_separation(z_np, labels, primes):
    centroids = {p: z_np[labels == p].mean(axis=0) for p in primes}
    within = np.mean([
        np.linalg.norm(z_np[labels == p] - centroids[p], axis=1).mean()
        for p in primes
    ])
    c_mat  = np.array([centroids[p] for p in primes])
    diffs  = c_mat[:, None] - c_mat[None, :]
    between = np.linalg.norm(diffs, axis=-1)
    idx    = np.triu_indices(len(primes), k=1)
    between_mean = between[idx].mean()
    return between_mean / (within + 1e-8)   # > 1 means clusters separated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/hierarchical_3level_supcon')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--samples_per_type', type=int, default=SAMPLES)
    args = parser.parse_args()

    label = args.label or os.path.basename(args.checkpoint.rstrip('/'))
    print(f'\n=== Evaluating: {label} ===')
    print(f'Checkpoint: {args.checkpoint}')

    model = load_model(args.checkpoint)
    z_raw, z_proj, digits, p_tensor = extract_representations(
        model, PRIMES, args.samples_per_type)

    labels = p_tensor.numpy()
    z_raw_np  = z_raw.numpy()
    z_proj_np = z_proj.numpy()

    # 1. Spearman r per prime
    print('\n--- Spearman r (latent vs p-adic distance) ---')
    r_raw  = spearman_per_prime(z_raw,  digits, p_tensor, PRIMES)
    r_proj = spearman_per_prime(z_proj, digits, p_tensor, PRIMES)
    print(f"{'Prime':<8} {'raw r':>8} {'proj r':>8}")
    for p_val in PRIMES:
        print(f"p={p_val:<6} {r_raw.get(p_val, float('nan')):>8.4f} "
              f"{r_proj.get(p_val, float('nan')):>8.4f}")
    mean_raw  = np.mean(list(r_raw.values()))
    mean_proj = np.mean(list(r_proj.values()))
    print(f"{'Mean':<8} {mean_raw:>8.4f} {mean_proj:>8.4f}")

    # 2. kNN prime prediction accuracy
    print('\n--- kNN Prime Prediction Accuracy ---')
    for k in [1, 5]:
        acc_raw  = knn_prime_accuracy(z_raw_np,  labels, k=k)
        acc_proj = knn_prime_accuracy(z_proj_np, labels, k=k)
        print(f"  {k}-NN: raw={acc_raw*100:.1f}%  proj={acc_proj*100:.1f}%")

    # 3. Silhouette score
    print('\n--- Silhouette Score (prime clusters) ---')
    sil_raw  = silhouette_score(z_raw_np,  labels, metric='euclidean',
                                sample_size=min(1000, len(labels)))
    sil_proj = silhouette_score(z_proj_np, labels, metric='euclidean',
                                sample_size=min(1000, len(labels)))
    print(f"  raw:  {sil_raw:.4f}")
    print(f"  proj: {sil_proj:.4f}")
    print("  (range [-1,1]; higher = better separated by prime)")

    # 4. Centroid separation ratio
    print('\n--- Centroid Separation (between/within) ---')
    sep_raw  = centroid_separation(z_raw_np,  labels, PRIMES)
    sep_proj = centroid_separation(z_proj_np, labels, PRIMES)
    print(f"  raw:  {sep_raw:.4f}")
    print(f"  proj: {sep_proj:.4f}")
    print("  (>1 = centroids farther apart than avg within-cluster radius)")

    # Summary
    print(f'\n=== Summary: {label} ===')
    print(f"  Spearman r (proj):      {mean_proj:.4f}")
    print(f"  5-NN prime acc (proj):  {knn_prime_accuracy(z_proj_np, labels, k=5)*100:.1f}%")
    print(f"  Silhouette (proj):      {sil_proj:.4f}")
    print(f"  Centroid sep (proj):    {sep_proj:.4f}")


if __name__ == '__main__':
    main()
