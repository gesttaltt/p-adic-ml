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
    
    save_dir = './checkpoints/broad_p17'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('./plots/comparison_p17', exist_ok=True)
    
    # Primes for the new model
    primes_new = [2, 3, 5, 7, 11, 13, 17]
    
    # Vocab size needs to be 17 to support digits up to 16
    vocab_size_new = 17
    
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
    # 2. Training the Broad-17 Models
    # -------------------------------------------------------------
    vqvae_new_path = os.path.join(save_dir, 'vqvae.pt')
    prior_new_path = os.path.join(save_dir, 'prior.pt')
    beta_vae_new_path = os.path.join(save_dir, 'beta_vae_metric.pt')
    
    # Training VQ-VAE
    print("\n--- [Step 2] Training Broad-17 VQ-VAE ---")
    vqvae_new = ConditionalVQVAE(
        vocab_size=vocab_size_new,
        hidden_dim=64,
        codebook_size=64,
        latent_dim=32,
        N=N,
        cond_dim=16
    )
    vqvae_new = train_vqvae(vqvae_new, train_loader, val_loader, vqvae_epochs, lr, device)
    torch.save(vqvae_new.state_dict(), vqvae_new_path)
    
    # Training Prior
    print("\n--- [Step 3] Training Broad-17 Autoregressive Prior ---")
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
    print("\n--- [Step 4] Training Broad-17 Aligned Beta-VAE ---")
    beta_vae_new = ConditionalBetaVAE(
        vocab_size=vocab_size_new,
        hidden_dim=64,
        latent_dim=32,
        N=N,
        cond_dim=16
    )
    beta_vae_new = train_beta_vae_metric(beta_vae_new, train_loader, val_loader, vae_epochs, lr, beta, gamma, device)
    torch.save(beta_vae_new.state_dict(), beta_vae_new_path)
    
    # -------------------------------------------------------------
    # 3. Load Pre-trained Models
    # -------------------------------------------------------------
    print("\n--- [Step 5] Loading Pre-trained Models ---")
    # A. Restricted [2, 5]
    vqvae_r = ConditionalVQVAE(vocab_size=13, hidden_dim=64, codebook_size=64, latent_dim=32, N=N, cond_dim=16)
    prior_r = PriorGRU(codebook_size=64, latent_dim=32, cond_dim=16, hidden_size=128, num_layers=2)
    beta_vae_r = ConditionalBetaVAE(vocab_size=13, hidden_dim=64, latent_dim=32, N=N, cond_dim=16)
    
    vqvae_r.load_state_dict(torch.load('./checkpoints/restricted/vqvae.pt', map_location=device))
    prior_r.load_state_dict(torch.load('./checkpoints/restricted/prior.pt', map_location=device))
    beta_vae_r.load_state_dict(torch.load('./checkpoints/restricted/beta_vae_metric.pt', map_location=device))
    
    # B. Broad-11 [2, 3, 5, 7, 11]
    vqvae_b11 = ConditionalVQVAE(vocab_size=13, hidden_dim=64, codebook_size=64, latent_dim=32, N=N, cond_dim=16)
    prior_b11 = PriorGRU(codebook_size=64, latent_dim=32, cond_dim=16, hidden_size=128, num_layers=2)
    beta_vae_b11 = ConditionalBetaVAE(vocab_size=13, hidden_dim=64, latent_dim=32, N=N, cond_dim=16)
    
    vqvae_b11.load_state_dict(torch.load('./checkpoints/vqvae.pt', map_location=device))
    prior_b11.load_state_dict(torch.load('./checkpoints/prior.pt', map_location=device))
    beta_vae_b11.load_state_dict(torch.load('./checkpoints/beta_vae_metric.pt', map_location=device))
    
    # C. Broad-13 [2, 3, 5, 7, 11, 13]
    vqvae_b13 = ConditionalVQVAE(vocab_size=13, hidden_dim=64, codebook_size=64, latent_dim=32, N=N, cond_dim=16)
    prior_b13 = PriorGRU(codebook_size=64, latent_dim=32, cond_dim=16, hidden_size=128, num_layers=2)
    beta_vae_b13 = ConditionalBetaVAE(vocab_size=13, hidden_dim=64, latent_dim=32, N=N, cond_dim=16)
    
    vqvae_b13.load_state_dict(torch.load('./checkpoints/broad_p13/vqvae.pt', map_location=device))
    prior_b13.load_state_dict(torch.load('./checkpoints/broad_p13/prior.pt', map_location=device))
    beta_vae_b13.load_state_dict(torch.load('./checkpoints/broad_p13/beta_vae_metric.pt', map_location=device))
    
    # Set all models to eval
    model_configs = [
        ('restricted', vqvae_r, beta_vae_r, prior_r),
        ('broad_11', vqvae_b11, beta_vae_b11, prior_b11),
        ('broad_13', vqvae_b13, beta_vae_b13, prior_b13),
        ('broad_17', vqvae_new, beta_vae_new, prior_new)
    ]
    for _, vqvae_m, beta_vae_m, prior_m in model_configs:
        vqvae_m.to(device).eval()
        beta_vae_m.to(device).eval()
        prior_m.to(device).eval()
        
    # -------------------------------------------------------------
    # 4. Comparative Evaluation on target primes [2, 5]
    # -------------------------------------------------------------
    print("\n--- [Step 6] Running Four-Way Comparative Evaluation ---")
    eval_samples_per_prime = 200
    dataset_eval = PadicDataset(primes=[2, 5], N=N, num_samples_per_type=eval_samples_per_prime)
    eval_loader = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)
    
    metrics = {
        'restricted': {'vq_acc_p2': 0.0, 'vq_acc_p5': 0.0, 'vae_recon_p2': 0.0, 'vae_recon_p5': 0.0, 'vae_metric_p2': 0.0, 'vae_metric_p5': 0.0},
        'broad_11': {'vq_acc_p2': 0.0, 'vq_acc_p5': 0.0, 'vae_recon_p2': 0.0, 'vae_recon_p5': 0.0, 'vae_metric_p2': 0.0, 'vae_metric_p5': 0.0},
        'broad_13': {'vq_acc_p2': 0.0, 'vq_acc_p5': 0.0, 'vae_recon_p2': 0.0, 'vae_recon_p5': 0.0, 'vae_metric_p2': 0.0, 'vae_metric_p5': 0.0},
        'broad_17': {'vq_acc_p2': 0.0, 'vq_acc_p5': 0.0, 'vae_recon_p2': 0.0, 'vae_recon_p5': 0.0, 'vae_metric_p2': 0.0, 'vae_metric_p5': 0.0}
    }
    
    for model_name, vqvae_m, beta_vae_m, prior_m in model_configs:
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
                
                # VQ-VAE
                logits_vq, _, _ = vqvae_m(digits, p)
                preds_vq = torch.argmax(logits_vq, dim=-1)
                
                # Beta-VAE
                logits_vae, mu_vae, logvar_vae = beta_vae_m(digits, p)
                z_vae = beta_vae_m.reparameterize(mu_vae, logvar_vae)
                
                # Recon error
                B = digits.shape[0]
                recon_criterion = nn.CrossEntropyLoss(reduction='none')
                # Reshape depending on model's vocab_size
                V = vqvae_m.vocab_size
                recon_loss_flat = recon_criterion(logits_vae.reshape(-1, V), digits.reshape(-1))
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
        
        if len(p2_z_list) > 1:
            metrics[model_name]['vae_metric_p2'] = compute_metric_loss(torch.stack(p2_z_list), torch.stack(p2_digits_list), torch.full((len(p2_z_list),), 2, dtype=torch.long, device=device)).item()
        if len(p5_z_list) > 1:
            metrics[model_name]['vae_metric_p5'] = compute_metric_loss(torch.stack(p5_z_list), torch.stack(p5_digits_list), torch.full((len(p5_z_list),), 5, dtype=torch.long, device=device)).item()

        # Prior sampling
        gen_samples = 200
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
    print("\n" + "="*110)
    print(f"{'Metric':<25} | {'Restricted [2, 5]':<18} | {'Broad-11 [2..11]':<18} | {'Broad-13 [2..13]':<18} | {'Broad-17 [2..17]':<18}")
    print("-" * 110)
    for k in sorted(metrics['restricted'].keys()):
        print(f"{k:<25} | {metrics['restricted'][k]:18.5f} | {metrics['broad_11'][k]:18.5f} | {metrics['broad_13'][k]:18.5f} | {metrics['broad_17'][k]:18.5f}")
    print("="*110)
    
    # -------------------------------------------------------------
    # 5. Save Results Report
    # -------------------------------------------------------------
    report_path = './plots/comparison_p17/results_report_p17.md'
    with open(report_path, 'w') as f:
        f.write("# Four-Way Comparison: Restricted [2, 5], Broad-11, Broad-13, and Extended Broad-17 Models\n\n")
        f.write("This report evaluates the scaling effects of multi-task regularization in conditional p-adic models. We compare a restricted model (2 primes), two broad models (5 and 6 primes), and a newly extended broad model (7 primes, adding $p=17$).\n\n")
        
        f.write("## Evaluation Metrics Summary Table\n\n")
        f.write("| Evaluation Metric | Restricted Model [2, 5] | Broad-11 Model [2..11] | Broad-13 Model [2..13] | Broad-17 Model [2..17] |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for k in sorted(metrics['restricted'].keys()):
            val_r = metrics['restricted'][k]
            val_b11 = metrics['broad_11'][k]
            val_b13 = metrics['broad_13'][k]
            val_b17 = metrics['broad_17'][k]
            f.write(f"| `{k}` | {val_r:.5f} | {val_b11:.5f} | {val_b13:.5f} | {val_b17:.5f} |\n")

    print(f"Comparison report saved to {report_path}")
    
    # -------------------------------------------------------------
    # 6. Generate Plot: VQ-VAE Reconstruction Accuracy scaling
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=150)
    labels = ['2-adic digits', '5-adic digits']
    acc_r = [metrics['restricted']['vq_acc_p2'] * 100, metrics['restricted']['vq_acc_p5'] * 100]
    acc_b11 = [metrics['broad_11']['vq_acc_p2'] * 100, metrics['broad_11']['vq_acc_p5'] * 100]
    acc_b13 = [metrics['broad_13']['vq_acc_p2'] * 100, metrics['broad_13']['vq_acc_p5'] * 100]
    acc_b17 = [metrics['broad_17']['vq_acc_p2'] * 100, metrics['broad_17']['vq_acc_p5'] * 100]
    
    x = np.arange(len(labels))
    width = 0.20
    
    plt.bar(x - 1.5*width, acc_r, width, label='Restricted [2, 5]', color='#ff9800')
    plt.bar(x - 0.5*width, acc_b11, width, label='Broad-11 [2..11]', color='#2196f3')
    plt.bar(x + 0.5*width, acc_b13, width, label='Broad-13 [2..13]', color='#00bcd4')
    plt.bar(x + 1.5*width, acc_b17, width, label='Broad-17 [2..17] (New)', color='#4caf50')
    
    plt.ylabel('Digit Reconstruction Accuracy (%)', fontweight='bold')
    plt.title('Reconstruction Performance Scaling across Prime Sets (up to p=17)', fontsize=12, fontweight='bold')
    plt.xticks(x, labels, fontweight='bold')
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend()
    
    plot_acc_path = './plots/comparison_p17/vqvae_accuracy_scaling.png'
    plt.savefig(plot_acc_path, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------
    # 7. Generate Plot: Latent Space PCA projections comparison (4 columns)
    # -------------------------------------------------------------
    print("Generating Latent Space PCA plots comparison...")
    fig, axes = plt.subplots(2, 4, figsize=(24, 14), dpi=150)
    
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
        
        # We need a quick projection for each model
        for col_idx, (model_name, _, beta_vae_m, _) in enumerate(model_configs):
            with torch.no_grad():
                mu, _ = beta_vae_m.encode(digits_tensor, p_tensor)
                z = beta_vae_m.reparameterize(mu, torch.zeros_like(mu))
            z_2d = project_pca(z.cpu(), 2).numpy()
            
            ax = axes[row_idx, col_idx]
            sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=residues, cmap='tab20', s=15, alpha=0.8)
            ax.set_title(f"{model_name} ({p}-adic)\n(Metric Loss: {metrics[model_name][f'vae_metric_p{p}']:.5f})", fontsize=11, fontweight='bold')
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            ax.grid(True, alpha=0.3)
            fig.colorbar(sc, ax=ax)
            
    plt.suptitle("Scaling Latent Space Topology\nRestricted vs. Broad-11 vs. Broad-13 vs. Broad-17", fontsize=16, fontweight='bold')
    plot_latent_path = './plots/comparison_p17/latent_space_scaling.png'
    plt.savefig(plot_latent_path, bbox_inches='tight')
    plt.close()
    
    print("Scaling comparison finished successfully!")

if __name__ == "__main__":
    main()
