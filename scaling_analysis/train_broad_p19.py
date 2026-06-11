import sys, os; root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.extend([root_dir, os.path.join(root_dir, 'src')]); os.chdir(root_dir)
import os
import time
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# Imports from codebase
from dataset import PadicDataset
from models import ConditionalVQVAE, PriorGRU
from beta_vae import ConditionalBetaVAE
from metric_alignment import compute_metric_loss
from train import train_vqvae, train_prior
from train_metric import train_beta_vae_metric
from anomaly_detector import get_reconstruction_error
from visualize_latent import project_pca
from visualize import check_periodicity
from visualize_scaling_trees import generate_tree_plot
from poincare_embedding import generate_poincare_disk

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # -------------------------------------------------------------
    # Configurations
    # -------------------------------------------------------------
    N = 64
    hidden_dim = 256   # increase from 64 to test capacity scaling (improvement #1)
    vqvae_epochs = 12
    prior_epochs = 12
    vae_epochs = 15
    samples_per_type = 600
    batch_size = 128
    lr = 1e-3
    beta = 0.05
    gamma = 5.0

    save_dir = f'./checkpoints/broad_p19_hd{hidden_dim}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('./plots/comparison_p19', exist_ok=True)
    
    # Primes for the new model
    primes_new = [2, 3, 5, 7, 11, 13, 17, 19]
    vocab_size_new = 19
    
    # -------------------------------------------------------------
    # 1. Dataset generation
    # -------------------------------------------------------------
    print(f"\n--- [Step 1] Preparing Datasets for primes {primes_new} ---")
    dataset_new = PadicDataset(primes=primes_new, N=N, num_samples_per_type=samples_per_type)
    val_size = int(0.1 * len(dataset_new))
    train_size = len(dataset_new) - val_size
    train_ds, val_ds = random_split(dataset_new, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # -------------------------------------------------------------
    # 2. Training the Broad-19 Models
    # -------------------------------------------------------------
    vqvae_new_path = os.path.join(save_dir, 'vqvae.pt')
    prior_new_path = os.path.join(save_dir, 'prior.pt')
    beta_vae_new_path = os.path.join(save_dir, 'beta_vae_metric.pt')
    
    # Training VQ-VAE
    print("\n--- [Step 2] Training Broad-19 VQ-VAE ---")
    vqvae_new = ConditionalVQVAE(
        vocab_size=vocab_size_new,
        hidden_dim=hidden_dim,
        codebook_size=64,
        latent_dim=32,
        N=N,
        cond_dim=16,
    )
    vqvae_new = train_vqvae(vqvae_new, train_loader, val_loader, vqvae_epochs, lr, device)
    torch.save(vqvae_new.state_dict(), vqvae_new_path)
    
    # Training Prior
    print("\n--- [Step 3] Training Broad-19 Autoregressive Prior ---")
    prior_new = PriorGRU(
        codebook_size=64,
        latent_dim=32,
        cond_dim=16,
        hidden_size=128,
        num_layers=2
    )
    full_loader = DataLoader(dataset_new, batch_size=batch_size, shuffle=False)
    prior_new = train_prior(vqvae_new, prior_new, full_loader, prior_epochs, lr, device)
    torch.save(prior_new.state_dict(), prior_new_path)
    
    # Training Aligned Beta-VAE
    print("\n--- [Step 4] Training Broad-19 Aligned Beta-VAE ---")
    beta_vae_new = ConditionalBetaVAE(
        vocab_size=vocab_size_new,
        hidden_dim=hidden_dim,
        latent_dim=32,
        N=N,
        cond_dim=16,
    )
    beta_vae_new = train_beta_vae_metric(beta_vae_new, train_loader, val_loader, vae_epochs, lr, beta, gamma, device)
    torch.save(beta_vae_new.state_dict(), beta_vae_new_path)
    
    # -------------------------------------------------------------
    # 3. Evaluate newly trained Broad-19 hd=256 on target primes [2, 5]
    # (Old Restricted/Broad-11/13/17 checkpoints used the legacy categorical
    #  prime_emb and are architecture-incompatible with the current PrimeEmbedder.
    #  Their numbers are documented in the README from the original scaling runs.)
    # -------------------------------------------------------------
    model_configs = [
        ('broad_19_hd256', vqvae_new, beta_vae_new, prior_new)
    ]
    for _, vqvae_m, beta_vae_m, prior_m in model_configs:
        vqvae_m.to(device).eval()
        beta_vae_m.to(device).eval()
        prior_m.to(device).eval()

    # -------------------------------------------------------------
    # 4. Evaluation on target primes [2, 5]
    # -------------------------------------------------------------
    print("\n--- [Step 5] Evaluating Broad-19 hd=256 on p=2 and p=5 ---")
    eval_samples_per_prime = 200
    dataset_eval = PadicDataset(primes=[2, 5], N=N, num_samples_per_type=eval_samples_per_prime)
    eval_loader = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)

    metrics = {
        'broad_19_hd256': {'vq_acc_p2': 0.0, 'vq_acc_p5': 0.0, 'vae_metric_p2': 0.0, 'vae_metric_p5': 0.0}
    }

    for model_name, vqvae_m, beta_vae_m, prior_m in model_configs:
        p2_vq_correct, p2_vq_total = 0, 0
        p5_vq_correct, p5_vq_total = 0, 0
        
        p2_z_list = []
        p2_digits_list = []
        p5_z_list = []
        p5_digits_list = []
        
        with torch.no_grad():
            for batch in eval_loader:
                digits = batch['digits'].to(device)
                p = batch['p'].to(device)
                
                # VQ-VAE
                logits_vq, _, _ = vqvae_m(digits, p)
                preds_vq = torch.argmax(logits_vq, dim=-1)
                
                # Beta-VAE
                logits_vae, mu_vae, logvar_vae = beta_vae_m(digits, p)
                z_vae = beta_vae_m.reparameterize(mu_vae, logvar_vae)
                
                B = digits.shape[0]
                for i in range(B):
                    prime = p[i].item()
                    correct = (preds_vq[i] == digits[i]).sum().item()
                    total = N
                    
                    if prime == 2:
                        p2_vq_correct += correct
                        p2_vq_total += total
                        p2_z_list.append(z_vae[i])
                        p2_digits_list.append(digits[i])
                    elif prime == 5:
                        p5_vq_correct += correct
                        p5_vq_total += total
                        p5_z_list.append(z_vae[i])
                        p5_digits_list.append(digits[i])
                        
        metrics[model_name]['vq_acc_p2'] = p2_vq_correct / p2_vq_total
        metrics[model_name]['vq_acc_p5'] = p5_vq_correct / p5_vq_total
        
        if len(p2_z_list) > 1:
            metrics[model_name]['vae_metric_p2'] = compute_metric_loss(torch.stack(p2_z_list), torch.stack(p2_digits_list), torch.full((len(p2_z_list),), 2, dtype=torch.long, device=device)).item()
        if len(p5_z_list) > 1:
            metrics[model_name]['vae_metric_p5'] = compute_metric_loss(torch.stack(p5_z_list), torch.stack(p5_digits_list), torch.full((len(p5_z_list),), 5, dtype=torch.long, device=device)).item()

    # -------------------------------------------------------------
    # 5. Save Results Report
    # -------------------------------------------------------------
    m = metrics['broad_19_hd256']
    print(f"\nBroad-19 hd=256 results on p=2/5:")
    print(f"  VQ-VAE acc p=2:      {m['vq_acc_p2']*100:.2f}%")
    print(f"  VQ-VAE acc p=5:      {m['vq_acc_p5']*100:.2f}%")
    print(f"  Metric align p=2:    {m['vae_metric_p2']:.5f}")
    print(f"  Metric align p=5:    {m['vae_metric_p5']:.5f}")

    report_path = './plots/comparison_p19/results_report_p19.md'
    with open(report_path, 'w') as f:
        f.write("# Broad-19 hd=256 Evaluation\n\n")
        f.write("| Metric | Broad-19 hd=256 |\n")
        f.write("| :--- | :---: |\n")
        for k in sorted(m.keys()):
            f.write(f"| `{k}` | {m[k]:.5f} |\n")
    print(f"Report saved to {report_path}")

    # -------------------------------------------------------------
    # 6. Generate Plot: Latent Space PCA (Broad-19 hd=256)
    # -------------------------------------------------------------
    print("Generating Latent Space PCA plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    for col_idx, p_eval in enumerate([2, 5]):
        ds_p = PadicDataset(primes=[p_eval], N=N, num_samples_per_type=200)
        digits_list, p_list, residues = [], [], []
        for sample in ds_p:
            if sample['type'] != 2:
                digits_list.append(sample['digits'])
                p_list.append(sample['p'])
                residues.append(sample['digits'][0].item() + sample['digits'][1].item() * p_eval)
        digits_tensor = torch.stack(digits_list).to(device)
        p_tensor = torch.tensor(p_list, dtype=torch.long, device=device)
        with torch.no_grad():
            mu, _ = beta_vae_new.encode(digits_tensor, p_tensor)
            z = beta_vae_new.reparameterize(mu, torch.zeros_like(mu))
        z_2d = project_pca(z.cpu(), 2).numpy()
        ax = axes[col_idx]
        sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=np.array(residues), cmap='tab20', s=15, alpha=0.8)
        ax.set_title(f"Broad-19 hd=256 ({p_eval}-adic)\nMetric Loss: {metrics['broad_19_hd256'][f'vae_metric_p{p_eval}']:.5f}", fontweight='bold')
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax)
    plt.suptitle("Latent Space PCA — Broad-19 hd=256", fontsize=14, fontweight='bold')
    plot_latent_path = './plots/comparison_p19/latent_space_scaling.png'
    plt.savefig(plot_latent_path, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 7. Generate 19-adic tree and Poincaré Disk
    # -------------------------------------------------------------
    generate_tree_plot(
        vqvae_path=vqvae_new_path,
        prior_path=prior_new_path,
        p=19,
        vocab_size=19,
        N=64,
        save_path='./plots/padic_tree_19.png',
        device=device,
        hidden_dim=hidden_dim,
    )

    generate_poincare_disk(
        vqvae_path=vqvae_new_path,
        prior_path=prior_new_path,
        p=19,
        vocab_size=19,
        save_path='./plots/poincare_p19.png',
        device=device,
        hidden_dim=hidden_dim,
    )
    
    # Copy to artifacts directory if environment variable is set and exists
    artifacts_dir = os.environ.get('ARTIFACTS_DIR')
    if artifacts_dir and os.path.exists(artifacts_dir):
        os.system(f"cp ./plots/comparison_p19/*.png {artifacts_dir}/")
        os.system(f"cp ./plots/padic_tree_19.png {artifacts_dir}/")
        os.system(f"cp ./plots/poincare_p19.png {artifacts_dir}/")
    
    print("Scaling comparison for p=19 finished successfully!")

if __name__ == "__main__":
    main()
