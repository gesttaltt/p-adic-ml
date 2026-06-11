import sys, os; root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.extend([root_dir, os.path.join(root_dir, 'src')]); os.chdir(root_dir)
import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import ConditionalVQVAE, PriorGRU
from dataset import PadicDataset

def get_poincare_coords(digits, p, max_depth=16, c=0.25):
    """
    Map a sequence of digits to Poincaré disk coordinates (x, y) at each depth.
    digits: list of digits
    p: prime base
    max_depth: max length to embed
    c: scaling parameter for hyperbolic distance (radius r = tanh(c * depth))
    """
    digits = digits[:max_depth]
    
    # Root is at (0, 0)
    x = [0.0]
    y = [0.0]
    
    # Initialize angular range for partitioning
    theta_min = 0.0
    theta_max = 2 * math.pi
    
    for depth, d in enumerate(digits):
        # Hyperbolic radius at this depth
        r = math.tanh(c * (depth + 1))
        
        # Partition angular sector
        sector_width = (theta_max - theta_min) / p
        theta_min = theta_min + d * sector_width
        theta_max = theta_min + sector_width
        
        # Angle is the midpoint of the sector
        theta = (theta_min + theta_max) / 2.0
        
        # Convert polar to Cartesian
        x.append(r * math.cos(theta))
        y.append(r * math.sin(theta))
        
    return x, y

def generate_poincare_disk(vqvae_path, prior_path, p, vocab_size, N=64, save_path='./plots/poincare.png', device='cpu', hidden_dim=64):
    print(f"Generating Poincaré disk for {p}-adic numbers...")

    # 1. Load Models
    vqvae = ConditionalVQVAE(vocab_size=vocab_size, hidden_dim=hidden_dim, codebook_size=64, latent_dim=32, N=N)
    vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
    vqvae.to(device).eval()

    prior = PriorGRU(codebook_size=64, latent_dim=32, cond_dim=16)
    prior.load_state_dict(torch.load(prior_path, map_location=device))
    prior.to(device).eval()
    
    # 2. Get evaluation dataset for comparison
    ds = PadicDataset(primes=[p], N=N, num_samples_per_type=20)
    real_rats = [s['digits'].tolist() for s in ds if s['type'] == 0]
    real_algs = [s['digits'].tolist() for s in ds if s['type'] == 1]
    
    # 3. Sample generated paths from the prior
    num_generate = 30
    p_tensor = torch.full((num_generate,), p, dtype=torch.long, device=device)
    with torch.no_grad():
        latent_indices = prior.sample(p_tensor, L=N // 2, temperature=0.7)
        quantized = vqvae.quantizer.embedding(latent_indices)
        logits = vqvae.decode(quantized, p_tensor)
        generated_digits = torch.argmax(logits, dim=-1).cpu().numpy()
        
    # 4. Plotting Poincaré Disk
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    
    # Draw unit circle boundary
    boundary = plt.Circle((0, 0), 1.0, color='grey', fill=False, linestyle='--', linewidth=1.5, alpha=0.8)
    ax.add_patch(boundary)
    
    # Draw background p-ary tree lines (limit depth dynamically to keep clean and fast)
    max_bg_depth = 5
    if p >= 13:
        max_bg_depth = 2
    elif p >= 5:
        max_bg_depth = 3
        
    def plot_background_tree(digits, depth, t_min, t_max):
        if depth >= max_bg_depth:
            return
        r_parent = math.tanh(c_factor * depth)
        r_child = math.tanh(c_factor * (depth + 1))
        
        sector_width = (t_max - t_min) / p
        for d in range(p):
            curr_t_min = t_min + d * sector_width
            curr_t_max = curr_t_min + sector_width
            
            theta_parent = (t_min + t_max) / 2.0 if depth > 0 else 0
            theta_child = (curr_t_min + curr_t_max) / 2.0
            
            x_p, y_p = r_parent * math.cos(theta_parent), r_parent * math.sin(theta_parent)
            x_c, y_c = r_child * math.cos(theta_child), r_child * math.sin(theta_child)
            
            if depth == 0:
                x_p, y_p = 0.0, 0.0
                
            ax.plot([x_p, x_c], [y_p, y_c], color='lightgrey', alpha=0.4, linewidth=1.0, zorder=1)
            plot_background_tree(digits + [d], depth + 1, curr_t_min, curr_t_max)

    c_factor = 0.3
    plot_background_tree([], 0, 0.0, 2 * math.pi)
    
    # Plot real rationals in blue (converging periodically to the boundary)
    for seq in real_rats[:10]:
        x, y = get_poincare_coords(seq, p, max_depth=16, c=c_factor)
        ax.plot(x, y, color='dodgerblue', alpha=0.6, linewidth=1.5, zorder=2)
        ax.scatter(x[-1], y[-1], color='blue', s=15, alpha=0.7, zorder=3)
        
    # Plot real algebraic roots in red
    for seq in real_algs[:10]:
        x, y = get_poincare_coords(seq, p, max_depth=16, c=c_factor)
        ax.plot(x, y, color='crimson', alpha=0.6, linewidth=1.5, zorder=2)
        ax.scatter(x[-1], y[-1], color='red', s=15, alpha=0.7, zorder=3)
        
    # Plot generated sequences in green
    for seq in generated_digits[:15]:
        x, y = get_poincare_coords(seq.tolist(), p, max_depth=16, c=c_factor)
        ax.plot(x, y, color='limegreen', alpha=0.7, linewidth=1.2, zorder=4)
        ax.scatter(x[-1], y[-1], color='darkgreen', s=15, alpha=0.8, zorder=5)
        
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title(f"Poincaré Disk Hyperbolic Embedding ({p}-adic Space)\nBlue = Rational, Red = Algebraic, Green = VQ-VAE Generated", fontsize=14, fontweight='bold')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Poincaré disk visualization to {save_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Primes to plot
    # Let's generate for p=2 (binary tree) and p=5 (5-ary tree)
    generate_poincare_disk(
        vqvae_path='./checkpoints/restricted/vqvae.pt',
        prior_path='./checkpoints/restricted/prior.pt',
        p=2,
        vocab_size=13,
        save_path='./plots/poincare_p2.png',
        device=device
    )
    
    generate_poincare_disk(
        vqvae_path='./checkpoints/restricted/vqvae.pt',
        prior_path='./checkpoints/restricted/prior.pt',
        p=5,
        vocab_size=13,
        save_path='./plots/poincare_p5.png',
        device=device
    )
    
    # Also generate for the broad models to see the higher bases: p=13 and p=17
    generate_poincare_disk(
        vqvae_path='./checkpoints/broad_p13/vqvae.pt',
        prior_path='./checkpoints/broad_p13/prior.pt',
        p=13,
        vocab_size=13,
        save_path='./plots/poincare_p13.png',
        device=device
    )
    
    generate_poincare_disk(
        vqvae_path='./checkpoints/broad_p17/vqvae.pt',
        prior_path='./checkpoints/broad_p17/prior.pt',
        p=17,
        vocab_size=17,
        save_path='./plots/poincare_p17.png',
        device=device
    )
    
    print("Poincaré disk plots generated successfully!")

if __name__ == "__main__":
    main()
