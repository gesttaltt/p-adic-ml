import sys, os; root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.extend([root_dir, os.path.join(root_dir, 'src')]); os.chdir(root_dir)
import os
import math
import torch
import matplotlib.pyplot as plt
from models import ConditionalVQVAE, PriorGRU
from dataset import PadicDataset
from visualize import get_path_coords

def generate_tree_plot(vqvae_path, prior_path, p, vocab_size, N=64, save_path='./plots/tree.png', device='cpu', hidden_dim=64):
    print(f"Generating {p}-adic tree visualization using model at {vqvae_path}...")

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
    num_generate = 50
    p_tensor = torch.full((num_generate,), p, dtype=torch.long, device=device)
    with torch.no_grad():
        latent_indices = prior.sample(p_tensor, L=N // 2, temperature=0.7)
        quantized = vqvae.quantizer.embedding(latent_indices)
        logits = vqvae.decode(quantized, p_tensor)
        generated_digits = torch.argmax(logits, dim=-1).cpu().numpy()
        
    # 4. Plotting the tree
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
        plt.plot(x, y, color='limegreen', alpha=0.6, linewidth=1.0, zorder=3)
        plt.scatter(x[-1], y[-1], color='forestgreen', s=10, alpha=0.8, zorder=4)
        
    plt.axis('equal')
    plt.axis('off')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved tree visualization to {save_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate 13-adic tree plot
    generate_tree_plot(
        vqvae_path='./checkpoints/broad_p13/vqvae.pt',
        prior_path='./checkpoints/broad_p13/prior.pt',
        p=13,
        vocab_size=13,
        N=64,
        save_path='./plots/padic_tree_13.png',
        device=device
    )
    
    # Generate 17-adic tree plot
    generate_tree_plot(
        vqvae_path='./checkpoints/broad_p17/vqvae.pt',
        prior_path='./checkpoints/broad_p17/prior.pt',
        p=17,
        vocab_size=17,
        N=64,
        save_path='./plots/padic_tree_17.png',
        device=device
    )
    
    print("Tree visualization plots generated successfully!")

if __name__ == "__main__":
    main()
