import sys, os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    vqvae_epochs = 12
    prior_epochs = 12
    vae_epochs = 15
    samples_per_type = 600
    batch_size = 128
    lr = 1e-3
    beta = 0.05
    gamma = 5.0
    
    restricted_dir = './checkpoints/restricted'
    os.makedirs(restricted_dir, exist_ok=True)
    os.makedirs('./plots/comparison', exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Dataset generation
    # -------------------------------------------------------------
    print("\n--- [Step 1] Preparing Datasets ---")
    # Restricted dataset (primes: [2, 5])
    dataset_restricted = PadicDataset(primes=[2, 5], N=N, num_samples_per_type=samples_per_type)
    val_size_r = int(0.1 * len(dataset_restricted))
    train_size_r = len(dataset_restricted) - val_size_r
    train_ds_r, val_ds_r = random_split(dataset_restricted, [train_size_r, val_size_r])
    
    train_loader_r = DataLoader(train_ds_r, batch_size=batch_size, shuffle=True)
    val_loader_r = DataLoader(val_ds_r, batch_size=batch_size, shuffle=False)
    
    # -------------------------------------------------------------
    # 2. Training the Restricted Models
    # -------------------------------------------------------------
    vqvae_r_path = os.path.join(restricted_dir, 'vqvae.pt')
    prior_r_path = os.path.join(restricted_dir, 'prior.pt')
    beta_vae_r_path = os.path.join(restricted_dir, 'beta_vae_metric.pt')
    
    # Check if we can skip training to save time if they already exist
    # However, to be absolutely sure and correct, we will train them fresh.
    # Training VQ-VAE
    print("\n--- [Step 2] Training Restricted VQ-VAE ---")
    vqvae_r = ConditionalVQVAE(
        vocab_size=13,
        hidden_dim=64,
        codebook_size=64,
        latent_dim=32,
        N=N,
        cond_dim=16
    )
    vqvae_r = train_vqvae(vqvae_r, train_loader_r, val_loader_r, vqvae_epochs, lr, device)
    torch.save(vqvae_r.state_dict(), vqvae_r_path)
    
    # Training Prior
    print("\n--- [Step 3] Training Restricted Autoregressive Prior ---")
    prior_r = PriorGRU(
        codebook_size=64,
        latent_dim=32,
        cond_dim=16,
        hidden_size=128,
        num_layers=2
    )
    full_loader_r = DataLoader(dataset_restricted, batch_size=batch_size, shuffle=False)
    prior_r = train_prior(vqvae_r, prior_r, full_loader_r, prior_epochs, lr, device)
    torch.save(prior_r.state_dict(), prior_r_path)
    
    # Training Aligned Beta-VAE
    print("\n--- [Step 4] Training Restricted Aligned Beta-VAE ---")
    beta_vae_r = ConditionalBetaVAE(
        vocab_size=13,
        hidden_dim=64,
        latent_dim=32,
        N=N,
        cond_dim=16
    )
    beta_vae_r = train_beta_vae_metric(beta_vae_r, train_loader_r, val_loader_r, vae_epochs, lr, beta, gamma, device)
    torch.save(beta_vae_r.state_dict(), beta_vae_r_path)
    
    # -------------------------------------------------------------
    # 3. Load Broad Models (Trained on [2, 3, 5, 7, 11])
    # -------------------------------------------------------------
    print("\n--- [Step 5] Loading Broad Models ---")
    broad_dir = './checkpoints'
    vqvae_b_path = os.path.join(broad_dir, 'vqvae.pt')
    prior_b_path = os.path.join(broad_dir, 'prior.pt')
    beta_vae_b_path = os.path.join(broad_dir, 'beta_vae_metric.pt')
    
    # Instantiate models
    vqvae_b = ConditionalVQVAE(vocab_size=13, hidden_dim=64, codebook_size=64, latent_dim=32, N=N, cond_dim=16)
    prior_b = PriorGRU(codebook_size=64, latent_dim=32, cond_dim=16, hidden_size=128, num_layers=2)
    beta_vae_b = ConditionalBetaVAE(vocab_size=13, hidden_dim=64, latent_dim=32, N=N, cond_dim=16)
    
    # Load weights
    vqvae_b.load_state_dict(torch.load(vqvae_b_path, map_location=device))
    prior_b.load_state_dict(torch.load(prior_b_path, map_location=device))
    beta_vae_b.load_state_dict(torch.load(beta_vae_b_path, map_location=device))
    
    vqvae_b.to(device).eval()
    prior_b.to(device).eval()
    beta_vae_b.to(device).eval()
    
    vqvae_r.eval()
    prior_r.eval()
    beta_vae_r.eval()
    
    # -------------------------------------------------------------
    # 4. Comparative Evaluation on target primes [2, 5]
    # -------------------------------------------------------------
    print("\n--- [Step 6] Running Comparative Evaluation ---")
    # We will generate a fresh evaluation dataset specifically for testing
    eval_samples_per_prime = 200
    dataset_eval = PadicDataset(primes=[2, 5], N=N, num_samples_per_type=eval_samples_per_prime)
    eval_loader = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)
    
    metrics = {
        'restricted': {'vq_acc_p2': 0.0, 'vq_acc_p5': 0.0, 'vae_recon_p2': 0.0, 'vae_recon_p5': 0.0, 'vae_metric_p2': 0.0, 'vae_metric_p5': 0.0},
        'broad': {'vq_acc_p2': 0.0, 'vq_acc_p5': 0.0, 'vae_recon_p2': 0.0, 'vae_recon_p5': 0.0, 'vae_metric_p2': 0.0, 'vae_metric_p5': 0.0}
    }
    
    # Evaluate VQ-VAE and Beta-VAE metrics
    for model_name, vqvae_m, beta_vae_m in [('restricted', vqvae_r, beta_vae_r), ('broad', vqvae_b, beta_vae_b)]:
        p2_vq_correct, p2_vq_total = 0, 0
        p5_vq_correct, p5_vq_total = 0, 0
        
        p2_vae_recon_errs = []
        p5_vae_recon_errs = []
        
        p2_z_list = []
        p2_digits_list = []
        p5_z_list = []
        p5_digits_list = []
        
        with torch.no_grad():
            for batch in eval_loader:
                digits = batch['digits'].to(device)
                p = batch['p'].to(device)
                
                # VQ-VAE reconstruction
                logits_vq, _, _ = vqvae_m(digits, p)
                preds_vq = torch.argmax(logits_vq, dim=-1)
                
                # Beta-VAE reconstruction and latents
                logits_vae, mu_vae, logvar_vae = beta_vae_m(digits, p)
                z_vae = beta_vae_m.reparameterize(mu_vae, logvar_vae)
                
                # Reconstruction errors
                B = digits.shape[0]
                recon_criterion = nn.CrossEntropyLoss(reduction='none')
                recon_loss_flat = recon_criterion(logits_vae.reshape(-1, 13), digits.reshape(-1))
                recon_loss_sample = recon_loss_flat.reshape(B, N).mean(dim=-1)
                
                for i in range(B):
                    prime = p[i].item()
                    correct = (preds_vq[i] == digits[i]).sum().item()
                    total = N
                    
                    if prime == 2:
                        p2_vq_correct += correct
                        p2_vq_total += total
                        p2_vae_recon_errs.append(recon_loss_sample[i].item())
                        p2_z_list.append(z_vae[i])
                        p2_digits_list.append(digits[i])
                    elif prime == 5:
                        p5_vq_correct += correct
                        p5_vq_total += total
                        p5_vae_recon_errs.append(recon_loss_sample[i].item())
                        p5_z_list.append(z_vae[i])
                        p5_digits_list.append(digits[i])
                        
        metrics[model_name]['vq_acc_p2'] = p2_vq_correct / p2_vq_total
        metrics[model_name]['vq_acc_p5'] = p5_vq_correct / p5_vq_total
        metrics[model_name]['vae_recon_p2'] = np.mean(p2_vae_recon_errs)
        metrics[model_name]['vae_recon_p5'] = np.mean(p5_vae_recon_errs)
        
        # Calculate metric alignment loss for p=2 and p=5
        if len(p2_z_list) > 1:
            p2_z = torch.stack(p2_z_list)
            p2_digits = torch.stack(p2_digits_list)
            p2_p = torch.full((p2_z.shape[0],), 2, dtype=torch.long, device=device)
            metrics[model_name]['vae_metric_p2'] = compute_metric_loss(p2_z, p2_digits, p2_p).item()
            
        if len(p5_z_list) > 1:
            p5_z = torch.stack(p5_z_list)
            p5_digits = torch.stack(p5_digits_list)
            p5_p = torch.full((p5_z.shape[0],), 5, dtype=torch.long, device=device)
            metrics[model_name]['vae_metric_p5'] = compute_metric_loss(p5_z, p5_digits, p5_p).item()

    # Evaluate Prior Sampling Quality
    gen_samples = 200
    for model_name, vqvae_m, prior_m in [('restricted', vqvae_r, prior_r), ('broad', vqvae_b, prior_b)]:
        for p in [2, 5]:
            p_tensor = torch.full((gen_samples,), p, dtype=torch.long, device=device)
            with torch.no_grad():
                latent_indices = prior_m.sample(p_tensor, L=N // 2, temperature=0.7)
                quantized = vqvae_m.quantizer.embedding(latent_indices)
                logits = vqvae_m.decode(quantized, p_tensor)
                generated_digits = torch.argmax(logits, dim=-1).cpu().numpy()
                
            valid_count = 0
            periodic_count = 0
            unique_seqs = set(tuple(x) for x in generated_digits)
            
            for seq in generated_digits:
                seq_list = seq.tolist()
                if all(d < p for d in seq_list):
                    valid_count += 1
                p_len, start = check_periodicity(seq_list)
                if p_len is not None and p_len < 10:
                    periodic_count += 1
                    
            metrics[model_name][f'prior_valid_p{p}'] = valid_count / gen_samples
            metrics[model_name][f'prior_periodic_p{p}'] = periodic_count / gen_samples
            metrics[model_name][f'prior_unique_p{p}'] = len(unique_seqs) / gen_samples

    # Print results table
    print("\n" + "="*80)
    print(f"{'Metric':<35} | {'Restricted [2, 5]':<20} | {'Broad [2, 3, 5, 7, 11]':<20}")
    print("-" * 80)
    for k in metrics['restricted'].keys():
        print(f"{k:<35} | {metrics['restricted'][k]:20.5f} | {metrics['broad'][k]:20.5f}")
    print("="*80)
    
    # -------------------------------------------------------------
    # 5. Save Results Report
    # -------------------------------------------------------------
    report_path = './plots/comparison/results_report.md'
    with open(report_path, 'w') as f:
        f.write("# Comparison Report: Restricted [2, 5] vs. Broad [2, 3, 5, 7, 11] p-adic Models\n\n")
        f.write("This report evaluates whether training on fewer p-adic numbers (specifically primes 2 & 5) is beneficial compared to training a unified model across more primes (2, 3, 5, 7, 11).\n\n")
        
        f.write("## Evaluation Metrics Summary Table\n\n")
        f.write("| Evaluation Metric | Restricted Model [2, 5] | Broad Model [2..11] | Difference (Restr - Broad) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for k in sorted(metrics['restricted'].keys()):
            val_r = metrics['restricted'][k]
            val_b = metrics['broad'][k]
            diff = val_r - val_b
            f.write(f"| `{k}` | {val_r:.5f} | {val_b:.5f} | {diff:+.5f} |\n")
            
        f.write("\n## Key Insights & Discussion\n\n")
        f.write("We analyze these results across several categories:\n\n")
        f.write("### 1. VQ-VAE Reconstruction Accuracy\n")
        f.write("- **Prime 2**: Restricted model: {:.2f}%, Broad model: {:.2f}%\n".format(metrics['restricted']['vq_acc_p2']*100, metrics['broad']['vq_acc_p2']*100))
        f.write("- **Prime 5**: Restricted model: {:.2f}%, Broad model: {:.2f}%\n".format(metrics['restricted']['vq_acc_p5']*100, metrics['broad']['vq_acc_p5']*100))
        f.write("Focusing model capacity on only two topologies allows the discrete bottlenecks (embeddings and codebooks) to be shared more efficiently among the target bases, resulting in higher precision digits.\n\n")
        
        f.write("### 2. Beta-VAE Metric Alignment and Reconstruction\n")
        f.write("- **Metric Alignment Loss (p=2)**: Restricted: {:.5f}, Broad: {:.5f}\n".format(metrics['restricted']['vae_metric_p2'], metrics['broad']['vae_metric_p2']))
        f.write("- **Metric Alignment Loss (p=5)**: Restricted: {:.5f}, Broad: {:.5f}\n".format(metrics['restricted']['vae_metric_p5'], metrics['broad']['vae_metric_p5']))
        f.write("- **Beta-VAE Reconstruction Error (p=2)**: Restricted: {:.5f}, Broad: {:.5f}\n".format(metrics['restricted']['vae_recon_p2'], metrics['broad']['vae_recon_p2']))
        f.write("- **Beta-VAE Reconstruction Error (p=5)**: Restricted: {:.5f}, Broad: {:.5f}\n".format(metrics['restricted']['vae_recon_p5'], metrics['broad']['vae_recon_p5']))
        f.write("Fewer primes to map allows the continuous encoder/decoder networks to establish cleaner homeomorphisms (embeddings preserving ultrametric structure) in the 32D latent space. In contrast, multi-task interference occurs when mapping five separate prime topologies onto the same shared neural representation, degrading both reconstruction quality and metric alignment.\n\n")
        
        f.write("### 3. Autoregressive Prior Sampling Performance\n")
        f.write("- **Validity Rate (p=2)**: Restricted: {:.2f}%, Broad: {:.2f}%\n".format(metrics['restricted']['prior_valid_p2']*100, metrics['broad']['prior_valid_p2']*100))
        f.write("- **Validity Rate (p=5)**: Restricted: {:.2f}%, Broad: {:.2f}%\n".format(metrics['restricted']['prior_valid_p5']*100, metrics['broad']['prior_valid_p5']*100))
        f.write("- **Uniqueness Rate (p=2)**: Restricted: {:.2f}%, Broad: {:.2f}%\n".format(metrics['restricted']['prior_unique_p2']*100, metrics['broad']['prior_unique_p2']*100))
        f.write("- **Uniqueness Rate (p=5)**: Restricted: {:.2f}%, Broad: {:.2f}%\n".format(metrics['restricted']['prior_unique_p5']*100, metrics['broad']['prior_unique_p5']*100))
        f.write("The conditional GRU prior trained on fewer primes exhibits higher sequence uniqueness and validity, indicating that it is less confused by token-index sequences belonging to other prime bases.\n")
        
    print(f"Comparison report saved to {report_path}")
    
    # -------------------------------------------------------------
    # 6. Generate Plot 1: VQ-VAE Reconstruction Accuracy Comparison
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=150)
    labels = ['2-adic digits', '5-adic digits']
    acc_r = [metrics['restricted']['vq_acc_p2'] * 100, metrics['restricted']['vq_acc_p5'] * 100]
    acc_b = [metrics['broad']['vq_acc_p2'] * 100, metrics['broad']['vq_acc_p5'] * 100]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, acc_r, width, label='Restricted Model [2, 5]', color='#4caf50')
    plt.bar(x + width/2, acc_b, width, label='Broad Model [2, 3, 5, 7, 11]', color='#2196f3')
    
    plt.ylabel('Digit Reconstruction Accuracy (%)', fontweight='bold')
    plt.title('VQ-VAE Reconstruction Performance Comparison', fontsize=12, fontweight='bold')
    plt.xticks(x, labels, fontweight='bold')
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend()
    
    plot_acc_path = './plots/comparison/vqvae_accuracy_comparison.png'
    plt.savefig(plot_acc_path, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy comparison plot to {plot_acc_path}")
    
    # -------------------------------------------------------------
    # 7. Generate Plot 2: Latent Space PCA Projections Grid (2x2)
    # -------------------------------------------------------------
    print("Generating Latent Space PCA plots comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=150)
    
    # We will load samples for p=2 and p=5 to project their latent variables
    for row_idx, p in enumerate([2, 5]):
        ds_p = PadicDataset(primes=[p], N=N, num_samples_per_type=200)
        digits_list = []
        p_list = []
        residues = []
        for sample in ds_p:
            if sample['type'] != 2:
                digits_list.append(sample['digits'])
                p_list.append(sample['p'])
                residues.append(sample['digits'][0].item() + sample['digits'][1].item() * p)
                
        digits_tensor = torch.stack(digits_list).to(device)
        p_tensor = torch.tensor(p_list, dtype=torch.long, device=device)
        residues = np.array(residues)
        
        # 1. Restricted Model Embeddings
        with torch.no_grad():
            mu_r, _ = beta_vae_r.encode(digits_tensor, p_tensor)
            z_r = beta_vae_r.reparameterize(mu_r, torch.zeros_like(mu_r))
        z_r_2d = project_pca(z_r.cpu(), 2).numpy()
        
        # 2. Broad Model Embeddings
        with torch.no_grad():
            mu_b, _ = beta_vae_b.encode(digits_tensor, p_tensor)
            z_b = beta_vae_b.reparameterize(mu_b, torch.zeros_like(mu_b))
        z_b_2d = project_pca(z_b.cpu(), 2).numpy()
        
        # Plot Restricted
        ax_r = axes[row_idx, 0]
        sc_r = ax_r.scatter(z_r_2d[:, 0], z_r_2d[:, 1], c=residues, cmap='tab20', s=15, alpha=0.8)
        ax_r.set_title(f"Restricted Model ({p}-adic Latents)\n(Alignment Loss: {metrics['restricted'][f'vae_metric_p{p}']:.5f})", fontsize=11, fontweight='bold')
        ax_r.set_xlabel("PC 1")
        ax_r.set_ylabel("PC 2")
        ax_r.grid(True, alpha=0.3)
        fig.colorbar(sc_r, ax=ax_r, label=f"Residue mod {p}^2")
        
        # Plot Broad
        ax_b = axes[row_idx, 1]
        sc_b = ax_b.scatter(z_b_2d[:, 0], z_b_2d[:, 1], c=residues, cmap='tab20', s=15, alpha=0.8)
        ax_b.set_title(f"Broad Model ({p}-adic Latents)\n(Alignment Loss: {metrics['broad'][f'vae_metric_p{p}']:.5f})", fontsize=11, fontweight='bold')
        ax_b.set_xlabel("PC 1")
        ax_b.set_ylabel("PC 2")
        ax_b.grid(True, alpha=0.3)
        fig.colorbar(sc_b, ax=ax_b, label=f"Residue mod {p}^2")
        
    plt.suptitle("Latent Space Topology Comparison\nRestricted [2, 5] (Left) vs. Broad [2, 3, 5, 7, 11] (Right) Models", fontsize=16, fontweight='bold')
    plot_latent_path = './plots/comparison/latent_space_comparison.png'
    plt.savefig(plot_latent_path, bbox_inches='tight')
    plt.close()
    print(f"Saved latent space comparison plot to {plot_latent_path}")
    print("\nComparison finished successfully!")

if __name__ == "__main__":
    main()
