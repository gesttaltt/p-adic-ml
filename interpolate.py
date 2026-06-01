import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import PadicDataset
from beta_vae import ConditionalBetaVAE
from hyperbolic_vae import HyperbolicBetaVAE

def get_pca_projection_params(z):
    """
    Compute mean and projection matrix Vh for PCA.
    z: [B, D]
    """
    z_mean = torch.mean(z, dim=0, keepdim=True)
    z_centered = z - z_mean
    U, S, Vh = torch.linalg.svd(z_centered, full_matrices=False)
    return z_mean, Vh[:2] # Return mean and first 2 principal components

def project_new_points(z, z_mean, Vh_2):
    """
    Project new points using pre-computed PCA parameters.
    z: [B, D]
    """
    z_centered = z - z_mean
    return torch.matmul(z_centered, Vh_2.t())

def run_interpolation(aligned_path, p=5, N=64, num_steps=11, save_img_dir='./plots',
                      model_type='euclidean', manifold_type='poincare', hidden_dim=64, latent_dim=32, vocab_size=13):
    os.makedirs(save_img_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    if model_type == 'euclidean':
        model = ConditionalBetaVAE(vocab_size=vocab_size, hidden_dim=hidden_dim, latent_dim=latent_dim, N=N)
    else:
        model = HyperbolicBetaVAE(vocab_size=vocab_size, hidden_dim=hidden_dim, latent_dim=latent_dim, N=N, manifold=manifold_type)
    model.load_state_dict(torch.load(aligned_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. Load Dataset
    print(f"Loading {p}-adic dataset...")
    ds = PadicDataset(primes=[p], N=N, num_samples_per_type=200)
    
    # 3. Find two sequences with the same prefix (first 2 digits) but different later digits
    seq1, seq2 = None, None
    for i in range(len(ds)):
        for j in range(i+1, len(ds)):
            s1 = ds[i]['digits']
            s2 = ds[j]['digits']
            # Exclude random noise
            if ds[i]['type'] == 2 or ds[j]['type'] == 2:
                continue
            if s1[0] == s2[0] and s1[1] == s2[1]:
                if (s1[2:6] != s2[2:6]).any():
                    seq1 = s1
                    seq2 = s2
                    break
        if seq1 is not None:
            break
            
    if seq1 is None or seq2 is None:
        print("Error: Could not find two suitable sequences sharing a prefix.")
        return
        
    print(f"\nEndpoint 1: {seq1.tolist()}")
    print(f"Endpoint 2: {seq2.tolist()}")
    print(f"Common prefix (first 2 digits): {seq1[:2].tolist()}\n")
    
    # 4. Perform Latent Interpolation
    s1_t = seq1.unsqueeze(0).to(device)
    s2_t = seq2.unsqueeze(0).to(device)
    p_t = torch.tensor([p], dtype=torch.long, device=device)
    
    with torch.no_grad():
        if model_type == 'euclidean':
            mu1, _ = model.encode(s1_t, p_t)
            mu2, _ = model.encode(s2_t, p_t)
        else:
            mu1_tang, _ = model.encode(s1_t, p_t)
            mu2_tang, _ = model.encode(s2_t, p_t)
            mu1 = model.reparameterize(mu1_tang, torch.zeros_like(mu1_tang))
            mu2 = model.reparameterize(mu2_tang, torch.zeros_like(mu2_tang))
        
    t_vals = np.linspace(0, 1, num_steps)
    interpolated_digits = []
    z_path = []
    
    for t in t_vals:
        if model_type == 'euclidean':
            z_t = (1 - t) * mu1 + t * mu2
            z_eucl = z_t
        else:
            # Geodesic interpolation
            v = model.manifold.logmap(mu1, mu2)
            z_t = model.manifold.projx(model.manifold.expmap(mu1, t * v))
            z_eucl = model.manifold.logmap0(z_t)
            
        z_path.append(z_eucl)
        
        with torch.no_grad():
            logits = model.decode(z_t, p_t)
            decoded = torch.argmax(logits, dim=-1)[0].cpu().numpy()
            interpolated_digits.append((t, decoded.tolist()))
            
    # Print the path of sequences
    print(f"{'t':<5} | {'p-adic Sequence':<96} | Prefix OK?")
    print("-" * 115)
    for t, seq in interpolated_digits:
        prefix_ok = (seq[0] == seq1[0].item() and seq[1] == seq1[1].item())
        seq_str = " ".join(str(d) for d in seq)
        print(f"{t:5.2f} | {seq_str} | {str(prefix_ok):<9}")
        
    # 5. Extract all validation latents for background plotting
    background_digits = []
    background_p = []
    residues = []
    for sample in ds:
        if sample['type'] != 2:
            background_digits.append(sample['digits'])
            background_p.append(p)
            residues.append(sample['digits'][0].item() + sample['digits'][1].item() * p)
            
    bg_t = torch.stack(background_digits).to(device)
    bg_p_t = torch.tensor(background_p, dtype=torch.long, device=device)
    
    with torch.no_grad():
        if model_type == 'euclidean':
            mu_bg, _ = model.encode(bg_t, bg_p_t)
        else:
            mu_bg_tang, _ = model.encode(bg_t, bg_p_t)
            z_bg = model.reparameterize(mu_bg_tang, torch.zeros_like(mu_bg_tang))
            mu_bg = model.manifold.logmap0(z_bg)
            
    z_mean, Vh_2 = get_pca_projection_params(mu_bg.cpu())
    bg_2d = project_new_points(mu_bg.cpu(), z_mean, Vh_2).numpy()
    path_2d = project_new_points(torch.cat(z_path, dim=0).cpu(), z_mean, Vh_2).numpy()
    
    # 6. Plotting
    plt.figure(figsize=(10, 8), dpi=150)
    
    # Scatter plot background points (validation set)
    plt.scatter(
        bg_2d[:, 0], bg_2d[:, 1],
        c=residues, cmap='tab20', s=20, alpha=0.3, label='p-adic Validation Samples'
    )
    
    # Plot the interpolation path
    plt.plot(path_2d[:, 0], path_2d[:, 1], color='black', linewidth=2.0, zorder=3)
    plt.scatter(path_2d[:, 0], path_2d[:, 1], color='red', s=40, edgecolor='black', zorder=4, label='Interpolation Path z(t)')
    
    # Mark Endpoints
    plt.scatter(path_2d[0, 0], path_2d[0, 1], color='blue', s=120, edgecolor='black', zorder=5, label='Start: Endpoint 1')
    plt.scatter(path_2d[-1, 0], path_2d[-1, 1], color='magenta', s=120, edgecolor='black', zorder=5, label='End: Endpoint 2')
    
    # Annotate path direction
    plt.annotate('t=0 (Start)', (path_2d[0, 0], path_2d[0, 1]), textcoords="offset points", xytext=(10,10), ha='center', fontweight='bold')
    plt.annotate('t=1 (End)', (path_2d[-1, 0], path_2d[-1, 1]), textcoords="offset points", xytext=(10,10), ha='center', fontweight='bold')
    
    plt.title(f"Continuous Latent Space Interpolation ({model_type.capitalize()}): {p}-adic Tree Climbing\n(Start and End share prefix {seq1[:2].tolist()})")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = os.path.join(save_img_dir, f'latent_interpolation_p{p}_{model_type}_{manifold_type}.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"\nSaved interpolation visualization plot to {plot_path}")

def run_cross_prime_interpolation(
    model_path,
    p_start=2,
    p_end=5,
    N=32,
    num_steps=11,
    decode_with=5,
    save_img_dir='./plots',
    vocab_size=13,
    model_type='euclidean',
    manifold_type='poincare',
    hidden_dim=64,
    latent_dim=32,
):
    """
    Interpolate between a sequence from p_start-adic space and one from p_end-adic
    space inside a shared multi-prime latent space.
    """
    os.makedirs(save_img_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'euclidean':
        model = ConditionalBetaVAE(vocab_size=vocab_size, hidden_dim=hidden_dim, latent_dim=latent_dim, N=N)
    else:
        model = HyperbolicBetaVAE(vocab_size=vocab_size, hidden_dim=hidden_dim, latent_dim=latent_dim, N=N, manifold=manifold_type)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    print(f"\nLoading {p_start}-adic and {p_end}-adic sequences...")
    ds_start = PadicDataset(primes=[p_start], N=N, num_samples_per_type=100)
    ds_end   = PadicDataset(primes=[p_end],   N=N, num_samples_per_type=100)

    seq_start = next(s['digits'] for s in ds_start if s['type'] != 2)
    seq_end   = next(s['digits'] for s in ds_end   if s['type'] != 2)

    p_start_t    = torch.tensor([p_start],    dtype=torch.long, device=device)
    p_end_t      = torch.tensor([p_end],      dtype=torch.long, device=device)
    p_decode_t   = torch.tensor([decode_with], dtype=torch.long, device=device)

    with torch.no_grad():
        if model_type == 'euclidean':
            mu1, _ = model.encode(seq_start.unsqueeze(0).to(device), p_start_t)
            mu2, _ = model.encode(seq_end.unsqueeze(0).to(device),   p_end_t)
        else:
            mu1_tang, _ = model.encode(seq_start.unsqueeze(0).to(device), p_start_t)
            mu2_tang, _ = model.encode(seq_end.unsqueeze(0).to(device),   p_end_t)
            mu1 = model.reparameterize(mu1_tang, torch.zeros_like(mu1_tang))
            mu2 = model.reparameterize(mu2_tang, torch.zeros_like(mu2_tang))

    t_vals = np.linspace(0, 1, num_steps)
    path_digits = []
    z_path = []

    for t in t_vals:
        if model_type == 'euclidean':
            z_t = (1 - t) * mu1 + t * mu2
            z_eucl = z_t
        else:
            v = model.manifold.logmap(mu1, mu2)
            z_t = model.manifold.projx(model.manifold.expmap(mu1, t * v))
            z_eucl = model.manifold.logmap0(z_t)
            
        z_path.append(z_eucl)
        with torch.no_grad():
            logits = model.decode(z_t, p_decode_t)
            decoded = torch.argmax(logits, dim=-1)[0].cpu().numpy()
        path_digits.append((t, decoded.tolist()))

    print(f"\n{'t':>5} | Decoded sequence (p={decode_with})")
    print("-" * 70)
    for t, seq in path_digits:
        print(f"{t:>5.2f} | {' '.join(str(d) for d in seq)}")

    # Background samples for PCA context
    bg_digits, bg_p, bg_residues = [], [], []
    for ds, prime in [(ds_start, p_start), (ds_end, p_end)]:
        for s in ds:
            if s['type'] != 2:
                bg_digits.append(s['digits'])
                bg_p.append(prime)
                bg_residues.append(s['digits'][0].item() + s['digits'][1].item() * prime)

    bg_t = torch.stack(bg_digits).to(device)
    bg_p_t = torch.tensor(bg_p, dtype=torch.long, device=device)

    with torch.no_grad():
        if model_type == 'euclidean':
            mu_bg, _ = model.encode(bg_t, bg_p_t)
        else:
            mu_bg_tang, _ = model.encode(bg_t, bg_p_t)
            z_bg = model.reparameterize(mu_bg_tang, torch.zeros_like(mu_bg_tang))
            mu_bg = model.manifold.logmap0(z_bg)

    z_mean, Vh_2 = get_pca_projection_params(mu_bg.cpu())
    bg_2d   = project_new_points(mu_bg.cpu(), z_mean, Vh_2).numpy()
    path_2d = project_new_points(torch.cat(z_path, dim=0).cpu(), z_mean, Vh_2).numpy()

    colors = ['#4fc3f7' if p == p_start else '#ff8a65' for p in bg_p]

    plt.figure(figsize=(10, 8), dpi=150)
    for color, (px, py) in zip(colors, bg_2d):
        plt.scatter(px, py, c=color, s=12, alpha=0.3)

    plt.scatter([], [], c='#4fc3f7', s=20, label=f'{p_start}-adic samples')
    plt.scatter([], [], c='#ff8a65', s=20, label=f'{p_end}-adic samples')

    plt.plot(path_2d[:, 0], path_2d[:, 1], color='black', linewidth=2.0, zorder=3)
    plt.scatter(path_2d[:, 0], path_2d[:, 1], color='red', s=40,
                edgecolor='black', zorder=4, label='Interpolation path z(t)')
    plt.scatter(path_2d[0, 0],  path_2d[0, 1],  color='blue',    s=120,
                edgecolor='black', zorder=5, label=f'z(0): {p_start}-adic start')
    plt.scatter(path_2d[-1, 0], path_2d[-1, 1], color='magenta', s=120,
                edgecolor='black', zorder=5, label=f'z(1): {p_end}-adic end')

    plt.title(
        f"Cross-Prime Interpolation ({model_type.capitalize()}): {p_start}-adic → {p_end}-adic\n"
        f"(decoded with p={decode_with})"
    )
    plt.xlabel("PC 1"); plt.ylabel("PC 2")
    plt.grid(True, alpha=0.3); plt.legend()

    plot_path = os.path.join(save_img_dir, f'cross_prime_interp_p{p_start}_to_p{p_end}_{model_type}_{manifold_type}.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"\nSaved cross-prime interpolation plot to {plot_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/euclidean_n64/beta_vae_metric.pt')
    parser.add_argument('--model_type', type=str, choices=['euclidean', 'hyperbolic'], default='euclidean')
    parser.add_argument('--manifold', type=str, choices=['poincare', 'lorentz'], default='poincare')
    parser.add_argument('--p', type=int, default=5, help='Prime base for single-prime interpolation')
    parser.add_argument('--N', type=int, default=64, help='Sequence length')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--vocab_size', type=int, default=13)
    parser.add_argument('--cross_prime', action='store_true', help='Run cross-prime interpolation instead')
    parser.add_argument('--p_start', type=int, default=2)
    parser.add_argument('--p_end', type=int, default=5)
    parser.add_argument('--decode_with', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='./plots')
    args = parser.parse_args()

    if args.cross_prime:
        run_cross_prime_interpolation(
            model_path=args.model_path,
            p_start=args.p_start,
            p_end=args.p_end,
            N=args.N,
            decode_with=args.decode_with,
            save_img_dir=args.save_dir,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            manifold_type=args.manifold,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim
        )
    else:
        run_interpolation(
            aligned_path=args.model_path,
            p=args.p,
            N=args.N,
            save_img_dir=args.save_dir,
            model_type=args.model_type,
            manifold_type=args.manifold,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            vocab_size=args.vocab_size
        )

if __name__ == '__main__':
    main()
