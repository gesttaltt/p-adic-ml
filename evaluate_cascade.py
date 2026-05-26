import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

from dataset import PadicDataset
from models import ConditionalVQVAE, PriorGRU
from beta_vae import ConditionalBetaVAE
from anomaly_detector import CascadeRouter, get_reconstruction_error

def train_beta_vae(model, train_loader, val_loader, epochs, lr, beta, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    print("\n--- Training Conditional Beta-VAE ---")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_correct = 0
        total_tokens = 0
        
        for batch in train_loader:
            digits = batch['digits'].to(device) # [B, N]
            p = batch['p'].to(device) # [B]
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, mu, logvar = model(digits, p) # logits: [B, N, vocab_size]
            
            # Loss calculations
            B, N, C = logits.shape
            recon_loss = criterion(logits.reshape(-1, C), digits.reshape(-1)).mean()
            
            # KL Divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + beta * kl_loss
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item() * B
            total_recon += recon_loss.item() * B
            total_kl += kl_loss.item() * B
            
            # Accuracy
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == digits).sum().item()
            total_tokens += B * N
            
        train_loss = total_loss / len(train_loader.dataset)
        train_recon = total_recon / len(train_loader.dataset)
        train_kl = total_kl / len(train_loader.dataset)
        train_acc = total_correct / total_tokens
        
        # Validation
        model.eval()
        val_correct = 0
        val_tokens = 0
        with torch.no_grad():
            for batch in val_loader:
                digits = batch['digits'].to(device)
                p = batch['p'].to(device)
                logits, _, _ = model(digits, p)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == digits).sum().item()
                val_tokens += digits.shape[0] * digits.shape[1]
        val_acc = val_correct / val_tokens
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}) | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
    return model

def benchmark_cascade(router, test_primes, num_samples, device):
    """
    Evaluates different thresholds tau for CascadeRouter.
    Returns: list of thresholds, average velocity (samples/sec), precision (VQ-VAE recon accuracy), and routing rates.
    """
    # Thresholds to evaluate: from 0 (always VQ-VAE) to 10 (almost always Beta-VAE)
    thresholds = [0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 10.0]
    
    velocities = []
    precisions = []
    fast_rates = []
    
    # Prepare batch of primes for generation
    p_batch = []
    for p in test_primes:
        p_batch.extend([p] * (num_samples // len(test_primes)))
    p_tensor = torch.tensor(p_batch, dtype=torch.long, device=device)
    B = p_tensor.shape[0]
    
    print(f"\n--- Benchmarking Cascade Router on {B} generations ---")
    
    for tau in thresholds:
        # Generate with Cascade
        final_digits, routed_paths, beta_recon_err, elapsed = router.generate_cascade(p_tensor, threshold_tau=tau, device=device)
        
        # 1. Velocity (samples per second)
        velocity = B / elapsed
        velocities.append(velocity)
        
        # 2. Fast Path Selection Rate
        fast_rate = sum(routed_paths) / B
        fast_rates.append(fast_rate)
        
        # 3. Precision (evaluated by VQ-VAE reconstruction accuracy of generated samples)
        # Since VQ-VAE learned to map structured p-adics, if it reconstructs the generated samples well, they are high quality.
        with torch.no_grad():
            recon_logits, _, _ = router.vq_vae(final_digits, p_tensor)
            recon_digits = torch.argmax(recon_logits, dim=-1)
            precision = (recon_digits == final_digits).float().mean().item()
            precisions.append(precision)
            
        print(f"Threshold tau: {tau:5.2f} | Fast Path Rate: {fast_rate*100:5.1f}% | Velocity: {velocity:8.1f} smpl/s | VQ-VAE Recon Precision: {precision*100:6.2f}%")
        
    return thresholds, velocities, precisions, fast_rates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primes', type=int, nargs='+', default=[2, 3, 5, 7])
    parser.add_argument('--N', type=int, default=32)
    parser.add_argument('--samples_per_type', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--beta_vae_epochs', type=int, default=12)
    parser.add_argument('--beta', type=float, default=1.5, help='KL Divergence weight')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset & Loaders
    dataset = PadicDataset(primes=args.primes, N=args.N, num_samples_per_type=args.samples_per_type)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Instantiate and train Conditional Beta-VAE
    beta_vae = ConditionalBetaVAE(vocab_size=13, hidden_dim=64, latent_dim=32, N=args.N)
    beta_vae = train_beta_vae(beta_vae, train_loader, val_loader, args.beta_vae_epochs, args.lr, args.beta, device)
    
    # Save Beta-VAE weights
    beta_vae_path = os.path.join(args.save_dir, 'beta_vae.pt')
    torch.save(beta_vae.state_dict(), beta_vae_path)
    print(f"Saved Beta-VAE checkpoints to {beta_vae_path}")
    
    vqvae_path = os.path.join(args.save_dir, 'vqvae.pt')
    prior_path = os.path.join(args.save_dir, 'prior.pt')
    beta_vae_metric_path = os.path.join(args.save_dir, 'beta_vae_metric.pt')
    
    if not os.path.exists(vqvae_path) or not os.path.exists(prior_path):
        print(f"Error: Could not find VQ-VAE checkpoints at {vqvae_path} or {prior_path}. Please train VQ-VAE first!")
        return
        
    vq_vae = ConditionalVQVAE(vocab_size=13, hidden_dim=64, codebook_size=64, latent_dim=32, N=args.N)
    vq_vae.load_state_dict(torch.load(vqvae_path, map_location=device))
    vq_vae.to(device)
    
    prior = PriorGRU(codebook_size=64, latent_dim=32, cond_dim=16)
    prior.load_state_dict(torch.load(prior_path, map_location=device))
    prior.to(device)
    
    # Load Aligned Beta-VAE
    beta_vae_aligned = ConditionalBetaVAE(vocab_size=13, hidden_dim=64, latent_dim=32, N=args.N)
    if os.path.exists(beta_vae_metric_path):
        beta_vae_aligned.load_state_dict(torch.load(beta_vae_metric_path, map_location=device))
        print(f"Loaded Aligned Beta-VAE from {beta_vae_metric_path}")
    else:
        print("Warning: beta_vae_metric.pt not found. Aligned Cascade comparison will be skipped.")
        beta_vae_aligned = None
        
    if beta_vae_aligned is not None:
        beta_vae_aligned.to(device)
    
    # 4. Setup Cascade Routers
    router_unaligned = CascadeRouter(beta_vae, vq_vae, prior)
    router_aligned = CascadeRouter(beta_vae_aligned, vq_vae, prior) if beta_vae_aligned is not None else None
    
    # 5. Measure Reconstruction Error distributions on Real Data
    print("\n--- Analyzing Reconstruction Error Distributions on Real Data ---")
    val_loader_full = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    beta_errors = []
    vqvae_errors = []
    
    for batch in val_loader_full:
        digits = batch['digits'].to(device)
        p = batch['p'].to(device)
        
        beta_err = get_reconstruction_error(beta_vae, digits, p, is_vqvae=False)
        vqvae_err = get_reconstruction_error(vq_vae, digits, p, is_vqvae=True)
        
        beta_errors.extend(beta_err.cpu().numpy())
        vqvae_errors.extend(vqvae_err.cpu().numpy())
        
    # Plot Error Distributions
    os.makedirs('./plots', exist_ok=True)
    plt.figure(figsize=(10, 5), dpi=150)
    plt.hist(beta_errors, bins=30, alpha=0.6, label='Beta-VAE Reconstruction Error', color='coral')
    plt.hist(vqvae_errors, bins=30, alpha=0.6, label='VQ-VAE Reconstruction Error', color='skyblue')
    plt.axvline(np.mean(beta_errors), color='orangered', linestyle='dashed', linewidth=1.5, label='Beta-VAE Mean')
    plt.axvline(np.mean(vqvae_errors), color='dodgerblue', linestyle='dashed', linewidth=1.5, label='VQ-VAE Mean')
    plt.title('Self-Reconstruction Error Distributions on Real Data (Unsupervised Anomaly Metric)')
    plt.xlabel('Cross-Entropy Reconstruction Error')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    error_plot_path = './plots/reconstruction_errors.png'
    plt.savefig(error_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved reconstruction error distribution plot to {error_plot_path}")
    
    # 6. Benchmark Cascade Routers
    print("\nBenchmarking Unaligned Cascade...")
    thresholds, vels_un, precs_un, rates_un = benchmark_cascade(router_unaligned, args.primes, num_samples=200, device=device)
    
    if router_aligned is not None:
        print("\nBenchmarking Aligned Cascade...")
        _, vels_al, precs_al, rates_al = benchmark_cascade(router_aligned, args.primes, num_samples=200, device=device)
    
    # Plot Cascade Trade-off Curves Comparison
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    
    color = 'tab:green'
    ax1.set_xlabel('Anomaly Detection Threshold (tau)')
    ax1.set_ylabel('Generation Velocity (samples/sec)', color=color)
    line1 = ax1.plot(thresholds, vels_un, marker='o', linestyle='--', color='lightgreen', label='Velocity (Unaligned Cascade)')
    lines = line1
    if router_aligned is not None:
        line1_al = ax1.plot(thresholds, vels_al, marker='o', color='forestgreen', label='Velocity (Aligned Cascade)')
        lines += line1_al
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Precision (VQ-VAE Reconstruction Accuracy %)', color=color)
    line2 = ax2.plot(thresholds, [p * 100 for p in precs_un], marker='s', linestyle=':', color='skyblue', label='Precision (Unaligned Cascade)')
    lines += line2
    if router_aligned is not None:
        line2_al = ax2.plot(thresholds, [p * 100 for p in precs_al], marker='s', color='dodgerblue', label='Precision (Aligned Cascade)')
        lines += line2_al
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add legends
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left')
    
    plt.title('Cascade Router Comparison: Aligned vs. Unaligned Beta-VAE\n(Precision-Velocity Trade-off curves)')
    comparison_plot_path = './plots/cascade_tradeoff_comparison.png'
    plt.savefig(comparison_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved cascade comparison plot to {comparison_plot_path}")

if __name__ == "__main__":
    main()
