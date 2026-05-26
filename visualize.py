import math
import os
import random
import torch
import matplotlib.pyplot as plt
from models import ConditionalVQVAE, PriorGRU
from dataset import PadicDataset
from padic_math import padic_to_float

def get_path_coords(digits, p, max_depth=8, r=0.75, initial_spread=math.pi*0.6):
    """
    Compute 2D coordinates for a p-adic number represented as a path down a p-ary tree.
    digits: list of integers in {0, ..., p-1}
    p: the prime base
    max_depth: depth of the tree to plot
    r: geometric contraction factor for branch length
    initial_spread: the opening angle of the tree at the root
    """
    digits = digits[:max_depth]
    
    # Root is at (0, 0)
    x = [0.0]
    y = [0.0]
    
    curr_x, curr_y = 0.0, 0.0
    curr_theta = math.pi / 2  # Facing upwards
    spread = initial_spread
    
    for i, d in enumerate(digits):
        # Map digit d in [0, p-1] to a ratio in [-0.5, 0.5]
        if p > 1:
            ratio = (d - (p - 1) / 2.0) / (p / 2.0)
        else:
            ratio = 0.0
            
        theta = curr_theta + ratio * (spread / 2.0)
        length = r ** i
        
        next_x = curr_x + length * math.cos(theta)
        next_y = curr_y + length * math.sin(theta)
        
        x.append(next_x)
        y.append(next_y)
        
        curr_x, curr_y = next_x, next_y
        curr_theta = theta
        spread = spread * 0.75 # shrink the spread for the next level
        
    return x, y

def check_periodicity(digits):
    """
    Check if a sequence of digits is periodic and return the period length.
    Returns (period_len, start_idx) if periodic, or (None, None) otherwise.
    """
    n = len(digits)
    # Check for period lengths from 1 to n//2
    for p_len in range(1, n // 2 + 1):
        for start in range(n - 2 * p_len):
            pattern = digits[start : start + p_len]
            # Check if this pattern repeats until the end
            is_periodic = True
            for i in range(start, n - p_len, p_len):
                chunk = digits[i : i + p_len]
                # Pad pattern if we are at the end of the list and chunk is shorter
                compare_len = min(p_len, len(chunk))
                if chunk[:compare_len] != pattern[:compare_len]:
                    is_periodic = False
                    break
            if is_periodic and len(pattern) > 0:
                # Double check that pattern is not just a sub-repetition of a smaller period
                return p_len, start
    return None, None

def evaluate_and_plot(vqvae_path, prior_path, save_img_dir='./plots', N=32, device='cpu'):
    os.makedirs(save_img_dir, exist_ok=True)
    
    # 1. Load Models
    vqvae = ConditionalVQVAE(vocab_size=13, hidden_dim=64, codebook_size=64, latent_dim=32, N=N)
    vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
    vqvae.to(device)
    vqvae.eval()
    
    prior = PriorGRU(codebook_size=64, latent_dim=32, cond_dim=16)
    prior.load_state_dict(torch.load(prior_path, map_location=device))
    prior.to(device)
    prior.eval()
    
    # 2. Generate and Analyze for different primes
    test_primes = [2, 3, 5, 7]
    num_generate = 100
    
    for p in test_primes:
        print(f"\n--- Analyzing {p}-adic generation ---")
        
        # Create a tiny test dataset of real examples to compare paths
        ds = PadicDataset(primes=[p], N=N, num_samples_per_type=20)
        
        real_rats = [s['digits'].tolist() for s in ds if s['type'] == 0]
        real_algs = [s['digits'].tolist() for s in ds if s['type'] == 1]
        
        # Sample from prior
        p_tensor = torch.full((num_generate,), p, dtype=torch.long, device=device)
        latent_indices = prior.sample(p_tensor, L=16, temperature=0.7)
        
        # Decode latent indices
        with torch.no_grad():
            quantized = vqvae.quantizer.embedding(latent_indices) # [B, L, D]
            logits = vqvae.decode(quantized, p_tensor)
            generated_digits = torch.argmax(logits, dim=-1).cpu().numpy() # [B, N]
            
        # Analysis
        valid_count = 0
        periodic_count = 0
        total_recon_acc = 0
        
        # Verify reconstruction of real data to measure VQ-VAE health
        real_tensor = torch.tensor(real_rats + real_algs, dtype=torch.long, device=device)
        p_eval = torch.full((real_tensor.shape[0],), p, dtype=torch.long, device=device)
        with torch.no_grad():
            recon_logits, _, _ = vqvae(real_tensor, p_eval)
            recon_digits = torch.argmax(recon_logits, dim=-1)
            recon_acc = (recon_digits == real_tensor).float().mean().item()
            print(f"VQ-VAE Reconstruction Accuracy on Real Data: {recon_acc*100:.2f}%")
            
        unique_gen = set(tuple(x) for x in generated_digits)
        print(f"Generated {num_generate} sequences. Unique sequences: {len(unique_gen)}/{num_generate}")
        
        for seq in generated_digits:
            seq_list = seq.tolist()
            # 1. Validity check: all digits must be < p
            if all(d < p for d in seq_list):
                valid_count += 1
            
            # 2. Periodicity check
            p_len, start = check_periodicity(seq_list)
            if p_len is not None and p_len < 10:
                periodic_count += 1
                
        print(f"  Valid {p}-adic digits check: {valid_count}/{num_generate} ({valid_count/num_generate*100:.1f}%)")
        print(f"  Short-period (rational-like) sequences: {periodic_count}/{num_generate} ({periodic_count/num_generate*100:.1f}%)")
        
        # 3. Plotting the p-adic tree
        plt.figure(figsize=(10, 10), dpi=150)
        plt.title(f"{p}-adic Tree Structure & Generated Paths\n(Depth = 8, Red = Algebraic, Blue = Rational, Green = Generated)", fontsize=14)
        
        # Plot real rationals in blue
        for seq in real_rats[:15]:
            x, y = get_path_coords(seq, p, max_depth=8)
            plt.plot(x, y, color='dodgerblue', alpha=0.5, linewidth=1.5, zorder=2)
            
        # Plot real algebraic roots in red
        for seq in real_algs[:15]:
            x, y = get_path_coords(seq, p, max_depth=8)
            plt.plot(x, y, color='crimson', alpha=0.5, linewidth=1.5, zorder=2)
            
        # Plot generated sequences in green
        for seq in generated_digits[:30]:
            x, y = get_path_coords(seq.tolist(), p, max_depth=8)
            plt.plot(x, y, color='mediumspringgreen' if p==2 else 'limegreen', alpha=0.6, linewidth=1.0, zorder=3)
            # Mark endpoints with small dot
            plt.scatter(x[-1], y[-1], color='forestgreen', s=10, alpha=0.8, zorder=4)
            
        plt.axis('equal')
        plt.axis('off')
        
        # Save plot
        plot_path = os.path.join(save_img_dir, f'padic_tree_{p}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization plot to {plot_path}")

if __name__ == "__main__":
    evaluate_and_plot('./checkpoints/vqvae.pt', './checkpoints/prior.pt')
