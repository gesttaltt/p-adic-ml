import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import PadicDataset
from beta_vae import ConditionalBetaVAE

def project_pca(z, num_components=2):
    """
    Perform PCA projection on z using PyTorch SVD.
    z: [B, D] tensor
    Returns: [B, num_components] projected coordinates
    """
    z_mean = torch.mean(z, dim=0, keepdim=True)
    z_centered = z - z_mean
    # SVD: z_centered = U * diag(S) * Vh
    U, S, Vh = torch.linalg.svd(z_centered, full_matrices=False)
    # Project: z_projected = z_centered * Vh^T (first num_components vectors)
    # Vh shape: [min(B, D), D], Vh.t() shape: [D, min(B, D)]
    projected = torch.matmul(z_centered, Vh[:num_components].t())
    return projected

def evaluate_and_plot_latents(unaligned_path, aligned_path, save_img_dir='./plots', N=64):
    os.makedirs(save_img_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Models
    model_unaligned = ConditionalBetaVAE(vocab_size=13, hidden_dim=64, latent_dim=32, N=N)
    model_unaligned.load_state_dict(torch.load(unaligned_path, map_location=device))
    model_unaligned.to(device)
    model_unaligned.eval()
    
    model_aligned = ConditionalBetaVAE(vocab_size=13, hidden_dim=64, latent_dim=32, N=N)
    model_aligned.load_state_dict(torch.load(aligned_path, map_location=device))
    model_aligned.to(device)
    model_aligned.eval()
    
    # Primes to analyze
    primes_to_plot = [2, 3, 5, 7]
    
    for p in primes_to_plot:
        print(f"\nExtracting latents for {p}-adic numbers...")
        # Generate 400 test samples of rationals and algebraic roots for this prime
        ds = PadicDataset(primes=[p], N=N, num_samples_per_type=150)
        
        # Collect inputs
        digits_list = []
        p_list = []
        residues = []  # To color-code: represent a_0 + a_1 * p (first 2 digits)
        
        for sample in ds:
            # We exclude random noise for cleaner topological visualization
            if sample['type'] != 2:
                seq = sample['digits']
                digits_list.append(seq)
                p_list.append(sample['p'])
                # Calculate first 2 digits residue: a_0 + a_1 * p
                res = seq[0].item() + seq[1].item() * p
                residues.append(res)
                
        digits_tensor = torch.stack(digits_list).to(device)
        p_tensor = torch.tensor(p_list, dtype=torch.long, device=device)
        residues = np.array(residues)
        
        # Get Latent Representations
        with torch.no_grad():
            # Unaligned
            mu_un, logvar_un = model_unaligned.encode(digits_tensor, p_tensor)
            z_un = model_unaligned.reparameterize(mu_un, logvar_un)
            
            # Aligned
            mu_al, logvar_al = model_aligned.encode(digits_tensor, p_tensor)
            z_al = model_aligned.reparameterize(mu_al, logvar_al)
            
        # Project to 2D
        z_un_2d = project_pca(z_un.cpu(), 2).numpy()
        z_al_2d = project_pca(z_al.cpu(), 2).numpy()
        
        # Create Plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=150)
        
        # Plot 1: Unaligned Latent Space
        scatter1 = ax1.scatter(
            z_un_2d[:, 0], z_un_2d[:, 1],
            c=residues, cmap='tab20', s=15, alpha=0.8
        )
        ax1.set_title(f"Standard Beta-VAE Latent Space\n(Posterior Collapse / Blurry Representation)")
        ax1.set_xlabel("PC 1")
        ax1.set_ylabel("PC 2")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Aligned Latent Space
        scatter2 = ax2.scatter(
            z_al_2d[:, 0], z_al_2d[:, 1],
            c=residues, cmap='tab20', s=15, alpha=0.8
        )
        # Add colorbar to the second plot to show the residues
        cbar = fig.colorbar(scatter2, ax=ax2, label=f"Residue modulo {p}^2")
        ax2.set_title(f"Ultrametric Aligned Latent Space\n(Enforced d_latent ~ d_{p}-adic)")
        ax2.set_xlabel("PC 1")
        ax2.set_ylabel("PC 2")
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"Latent Topology Mapping: {p}-adic Integers modulo {p}^2", fontsize=16, fontweight='bold')
        
        # Save Plot
        plot_path = os.path.join(save_img_dir, f'latent_space_p{p}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved latent space topology plot to {plot_path}")

if __name__ == "__main__":
    evaluate_and_plot_latents('./checkpoints/beta_vae.pt', './checkpoints/beta_vae_metric.pt', N=64)
