import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import PadicDataset
from beta_vae import ConditionalBetaVAE
from visualize_latent import project_pca

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

def run_interpolation(aligned_path, p=5, N=32, num_steps=11, save_img_dir='./plots'):
    os.makedirs(save_img_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    model = ConditionalBetaVAE(vocab_size=13, hidden_dim=64, latent_dim=32, N=N)
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
        mu1, _ = model.encode(s1_t, p_t)
        mu2, _ = model.encode(s2_t, p_t)
        
    # Linear interpolation between latents
    t_vals = np.linspace(0, 1, num_steps)
    interpolated_digits = []
    z_path = []
    
    for t in t_vals:
        z_t = (1 - t) * mu1 + t * mu2
        z_path.append(z_t)
        
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
            background_p.append(sample['p'])
            residues.append(sample['digits'][0].item() + sample['digits'][1].item() * p)
            
    bg_digits_t = torch.stack(background_digits).to(device)
    bg_p_t = torch.tensor(background_p, dtype=torch.long, device=device)
    
    with torch.no_grad():
        mu_bg, _ = model.encode(bg_digits_t, bg_p_t)
        
    # Project to 2D using PCA
    z_mean, Vh_2 = get_pca_projection_params(mu_bg.cpu())
    bg_2d = project_new_points(mu_bg.cpu(), z_mean, Vh_2).numpy()
    
    # Project the path coordinates
    z_path_tensor = torch.cat(z_path, dim=0).cpu()
    path_2d = project_new_points(z_path_tensor, z_mean, Vh_2).numpy()
    
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
    
    plt.title(f"Continuous Latent Space Interpolation: {p}-adic Tree Climbing\n(Start and End share prefix {seq1[:2].tolist()})")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = os.path.join(save_img_dir, f'latent_interpolation_p{p}.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"\nSaved interpolation visualization plot to {plot_path}")

if __name__ == "__main__":
    run_interpolation('./checkpoints/beta_vae_metric.pt', N=64)
