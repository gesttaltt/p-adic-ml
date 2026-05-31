import argparse
import math
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
            
            B, N, C = logits.shape
            recon_loss_flat = criterion(logits.reshape(-1, C), digits.reshape(-1))
            recon_loss_sample = recon_loss_flat.reshape(B, N).mean(dim=-1)
            weights = torch.tensor([math.log(val.item()) + 1.0 for val in p], device=device)
            recon_loss = (recon_loss_sample * weights).mean()
            
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

def benchmark_cascade(router, test_primes, num_samples, device, weighted=False, alpha=1.5):
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
    
    print(f"\n--- Benchmarking Cascade Router (weighted={weighted}) on {B} generations ---")
    
    for tau in thresholds:
        # Generate with Cascade
        final_digits, routed_paths, beta_recon_err, elapsed = router.generate_cascade(
            p_tensor, threshold_tau=tau, device=device, weighted=weighted, alpha=alpha
        )
        
        # 1. Velocity (samples per second)
        velocity = B / elapsed
        velocities.append(velocity)
        
        # 2. Fast Path Selection Rate
        fast_rate = sum(routed_paths) / B
        fast_rates.append(fast_rate)
        
        # 3. Precision (evaluated by VQ-VAE reconstruction accuracy of generated samples)
        with torch.no_grad():
            recon_logits, _, _ = router.vq_vae(final_digits, p_tensor)
            recon_digits = torch.argmax(recon_logits, dim=-1)
            precision = (recon_digits == final_digits).float().mean().item()
            precisions.append(precision)
            
        print(f"Threshold tau: {tau:5.2f} | Fast Path Rate: {fast_rate*100:5.1f}% | Velocity: {velocity:8.1f} smpl/s | VQ-VAE Recon Precision: {precision*100:6.2f}%")
        
    return thresholds, velocities, precisions, fast_rates

def compute_adaptive_thresholds(beta_vae, val_loader, k=1.0, device='cpu', weighted=False, alpha=1.5):
    """
    Compute base-specific thresholds: tau_p = mu_p + k * sigma_p
    """
    errors_by_prime = {}
    beta_vae.eval()
    with torch.no_grad():
        for batch in val_loader:
            digits = batch['digits'].to(device)
            p = batch['p'].to(device)
            logits, _, _ = beta_vae(digits, p)
            recon_digits = torch.argmax(logits, dim=-1)
            errs = get_reconstruction_error(beta_vae, recon_digits, p, weighted=weighted, alpha=alpha)
            for i in range(p.shape[0]):
                prime = p[i].item()
                if prime not in errors_by_prime:
                    errors_by_prime[prime] = []
                errors_by_prime[prime].append(errs[i].item())
                
    thresholds_dict = {}
    for prime, err_list in errors_by_prime.items():
        arr = np.array(err_list)
        mu = np.mean(arr)
        std = np.std(arr)
        thresholds_dict[prime] = float(mu + k * std)
    return thresholds_dict

def benchmark_adaptive_cascade(router, val_loader, test_primes, num_samples, device, weighted=False, alpha=1.5):
    """
    Evaluates CascadeRouter using adaptive thresholds for different multipliers k.
    """
    k_vals = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    velocities = []
    precisions = []
    fast_rates = []
    
    # Prepare batch of primes for generation
    p_batch = []
    for p in test_primes:
        p_batch.extend([p] * (num_samples // len(test_primes)))
    p_tensor = torch.tensor(p_batch, dtype=torch.long, device=device)
    B = p_tensor.shape[0]
    
    print(f"\n--- Benchmarking Adaptive Cascade Router (weighted={weighted}, k std dev) on {B} generations ---")
    
    for k in k_vals:
        # Compute adaptive thresholds for this k
        thresholds_dict = compute_adaptive_thresholds(
            router.beta_vae, val_loader, k=k, device=device, weighted=weighted, alpha=alpha
        )
        
        # Generate with Cascade
        final_digits, routed_paths, beta_recon_err, elapsed = router.generate_cascade(
            p_tensor, threshold_tau=thresholds_dict, device=device, weighted=weighted, alpha=alpha
        )
        
        # Velocity
        velocity = B / elapsed
        velocities.append(velocity)
        
        # Fast Path Rate
        fast_rate = sum(routed_paths) / B
        fast_rates.append(fast_rate)
        
        # Precision
        with torch.no_grad():
            recon_logits, _, _ = router.vq_vae(final_digits, p_tensor)
            recon_digits = torch.argmax(recon_logits, dim=-1)
            precision = (recon_digits == final_digits).float().mean().item()
            precisions.append(precision)
            
        print(f"Multiplier k: {k:5.1f} | Fast Path Rate: {fast_rate*100:5.1f}% | Velocity: {velocity:8.1f} smpl/s | VQ-VAE Recon Precision: {precision*100:6.2f}%")
        
    return k_vals, velocities, precisions, fast_rates

def calibrate_threshold(router, val_loader, p_target, device='cpu', weighted=False, alpha=1.5):
    """
    Finds the optimal threshold tau that maximizes the fast-path routing rate
    subject to the constraint that VQ-VAE precision >= p_target.
    Evaluates on validation data.
    """
    router.beta_vae.eval()
    router.vq_vae.eval()
    router.prior.eval()
    
    # 1. Collect all validation samples
    all_digits = []
    all_primes = []
    for batch in val_loader:
        all_digits.append(batch['digits'])
        all_primes.append(batch['p'])
    all_digits = torch.cat(all_digits, dim=0).to(device)
    all_primes = torch.cat(all_primes, dim=0).to(device)
    
    B = all_digits.shape[0]
    
    # 2. Get Beta-VAE reconstructions and self-reconstruction errors
    with torch.no_grad():
        logits_beta, _, _ = router.beta_vae(all_digits, all_primes)
        x_beta = torch.argmax(logits_beta, dim=-1)
        errs = get_reconstruction_error(router.beta_vae, x_beta, all_primes, weighted=weighted, alpha=alpha)
        
    # 3. Sort errors to define candidate thresholds
    sorted_errs = sorted(list(errs.cpu().numpy()))
    step = max(1, len(sorted_errs) // 50)
    candidate_thresholds = [0.0] + [float(sorted_errs[i]) for i in range(0, len(sorted_errs), step)] + [sorted_errs[-1] + 0.1]
    
    # 4. Search for the maximum threshold that satisfies the constraint
    best_tau = 0.0
    best_rate = 0.0
    actual_precision = 1.0
    
    print(f"\nCalibrating threshold for target precision {p_target*100:.1f}% (weighted={weighted})...")
    for tau in candidate_thresholds:
        fast_mask = errs < tau
        fallback_indices = (~fast_mask).nonzero(as_tuple=True)[0]
        num_fallback = fallback_indices.shape[0]
        
        final_digits = x_beta.clone()
        if num_fallback > 0:
            p_fallback = all_primes[fallback_indices]
            with torch.no_grad():
                logits_vq, _, _ = router.vq_vae(all_digits[fallback_indices], p_fallback)
                x_vq = torch.argmax(logits_vq, dim=-1)
            final_digits[fallback_indices] = x_vq
            
        with torch.no_grad():
            recon_logits, _, _ = router.vq_vae(final_digits, all_primes)
            recon_digits = torch.argmax(recon_logits, dim=-1)
            precision = (recon_digits == final_digits).float().mean().item()
            
        rate = (B - num_fallback) / B
        if precision >= p_target:
            best_tau = tau
            best_rate = rate
            actual_precision = precision
        else:
            break
            
    print(f"  Calibrated: tau = {best_tau:.4f} | Est. Routing Rate = {best_rate*100:.1f}% | Est. Precision = {actual_precision*100:.2f}%")
    return best_tau

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primes', type=int, nargs='+', default=[2, 3, 5, 7, 11])
    parser.add_argument('--N', type=int, default=32)
    parser.add_argument('--samples_per_type', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--beta_vae_epochs', type=int, default=12)
    parser.add_argument('--beta', type=float, default=1.5, help='KL Divergence weight')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to write newly trained Beta-VAE checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory containing pre-trained vqvae.pt / prior.pt / beta_vae_metric.pt '
                             '(defaults to --save_dir if not set)')
    parser.add_argument('--vocab_size', type=int, default=13,
                        help='Vocabulary size for VQ-VAE and Beta-VAE — must match the checkpoint '
                             '(e.g. 19 for Broad-19, 23 for Broad-23)')
    args = parser.parse_args()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.save_dir
    
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
    beta_vae = ConditionalBetaVAE(vocab_size=args.vocab_size, hidden_dim=64, latent_dim=32, N=args.N)
    beta_vae = train_beta_vae(beta_vae, train_loader, val_loader, args.beta_vae_epochs, args.lr, args.beta, device)

    # Save Beta-VAE weights
    beta_vae_path = os.path.join(args.save_dir, 'beta_vae.pt')
    torch.save(beta_vae.state_dict(), beta_vae_path)
    print(f"Saved Beta-VAE checkpoints to {beta_vae_path}")

    vqvae_path        = os.path.join(args.checkpoint_dir, 'vqvae.pt')
    prior_path        = os.path.join(args.checkpoint_dir, 'prior.pt')
    beta_vae_metric_path = os.path.join(args.checkpoint_dir, 'beta_vae_metric.pt')

    if not os.path.exists(vqvae_path) or not os.path.exists(prior_path):
        print(f"Error: Could not find VQ-VAE checkpoints at {vqvae_path} or {prior_path}. "
              f"Train VQ-VAE first (train.py) or point --checkpoint_dir at an existing directory.")
        return

    vq_vae = ConditionalVQVAE(vocab_size=args.vocab_size, hidden_dim=64, codebook_size=64, latent_dim=32, N=args.N)
    vq_vae.load_state_dict(torch.load(vqvae_path, map_location=device))
    vq_vae.to(device)

    prior = PriorGRU(codebook_size=64, latent_dim=32, cond_dim=16)
    prior.load_state_dict(torch.load(prior_path, map_location=device))
    prior.to(device)

    # Load Aligned Beta-VAE
    beta_vae_aligned = ConditionalBetaVAE(vocab_size=args.vocab_size, hidden_dim=64, latent_dim=32, N=args.N)
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
        
        beta_err = get_reconstruction_error(beta_vae, digits, p)
        vqvae_err = get_reconstruction_error(vq_vae, digits, p)
        
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
    
    # 6. Run Constraint-Based Calibration
    if router_aligned is not None:
        print("\n=======================================================")
        print("         THRESHOLD CALIBRATION ON VALIDATION DATA      ")
        print("=======================================================")
        calibrate_threshold(router_aligned, val_loader_full, p_target=0.98, device=device, weighted=False)
        calibrate_threshold(router_aligned, val_loader_full, p_target=0.95, device=device, weighted=False)
        calibrate_threshold(router_aligned, val_loader_full, p_target=0.90, device=device, weighted=False)
        
        calibrate_threshold(router_aligned, val_loader_full, p_target=0.98, device=device, weighted=True, alpha=1.5)
        calibrate_threshold(router_aligned, val_loader_full, p_target=0.95, device=device, weighted=True, alpha=1.5)
        calibrate_threshold(router_aligned, val_loader_full, p_target=0.90, device=device, weighted=True, alpha=1.5)
        print("=======================================================\n")
        
    # 7. Benchmark Aligned Cascade Router under different gating methods
    if router_aligned is not None:
        # A. Global Standard
        print("\nBenchmarking Aligned Cascade (Global Standard)...")
        thresholds, vels_g_std, precs_g_std, _ = benchmark_cascade(
            router_aligned, args.primes, num_samples=200, device=device, weighted=False
        )
        
        # B. Global Ultrametric-Weighted
        print("\nBenchmarking Aligned Cascade (Global Ultrametric-Weighted)...")
        _, vels_g_w, precs_g_w, _ = benchmark_cascade(
            router_aligned, args.primes, num_samples=200, device=device, weighted=True, alpha=1.5
        )
        
        # C. Adaptive Standard
        print("\nBenchmarking Aligned Cascade (Adaptive Standard)...")
        k_vals, vels_a_std, precs_a_std, _ = benchmark_adaptive_cascade(
            router_aligned, val_loader_full, args.primes, num_samples=200, device=device, weighted=False
        )
        
        # D. Adaptive Ultrametric-Weighted
        print("\nBenchmarking Aligned Cascade (Adaptive Ultrametric-Weighted)...")
        _, vels_a_w, precs_a_w, _ = benchmark_adaptive_cascade(
            router_aligned, val_loader_full, args.primes, num_samples=200, device=device, weighted=True, alpha=1.5
        )
        
        # Plot Cascade Trade-off Curves Comparison (Pareto Frontier Style: Precision vs Velocity)
        plt.figure(figsize=(10, 6), dpi=150)
        
        # Plot curves
        plt.plot(vels_g_std, [p * 100 for p in precs_g_std], marker='x', linestyle='-', color='#4fc3f7', label='Global Standard Gating')
        plt.plot(vels_g_w, [p * 100 for p in precs_g_w], marker='s', linestyle='-', color='#0288d1', linewidth=2.5, label='Global Ultrametric Gating (Decay 1.5)')
        plt.plot(vels_a_std, [p * 100 for p in precs_a_std], marker='x', linestyle='--', color='#ffb74d', label='Adaptive Standard Gating')
        plt.plot(vels_a_w, [p * 100 for p in precs_a_w], marker='s', linestyle='--', color='#f57c00', label='Adaptive Ultrametric Gating')
        
        # Annotate selected points on the Global Ultrametric curve
        for target_tau, label in [(0.6, 'tau=0.6'), (1.0, 'tau=1.0'), (1.5, 'tau=1.5')]:
            if target_tau in thresholds:
                idx = thresholds.index(target_tau)
                x = vels_g_w[idx]
                y = precs_g_w[idx] * 100
                plt.annotate(
                    label, (x, y), textcoords="offset points", xytext=(-10, 15),
                    arrowprops=dict(arrowstyle="->", color='#0288d1', lw=0.8),
                    fontsize=9, fontweight='bold', color='#01579b'
                )
                
        # Annotate selected points on the Adaptive Ultrametric curve
        for target_k, label in [(-1.0, 'k=-1.0'), (0.0, 'k=0.0')]:
            if target_k in k_vals:
                idx = k_vals.index(target_k)
                x = vels_a_w[idx]
                y = precs_a_w[idx] * 100
                plt.annotate(
                    label, (x, y), textcoords="offset points", xytext=(10, -15),
                    arrowprops=dict(arrowstyle="->", color='#f57c00', lw=0.8),
                    fontsize=9, color='#e65100'
                )
                
        plt.xlabel('Generation Velocity (samples/sec)', fontsize=11, fontweight='bold')
        plt.ylabel('VQ-VAE Reconstruction Precision (%)', fontsize=11, fontweight='bold')
        plt.title('Advanced Cascade Router Gating Comparison (Aligned VAE)\n(Pareto Frontier of Standard CE vs. Ultrametric Error-Density Routing)', fontsize=13, fontweight='bold')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='lower left', framealpha=0.9)
        
        # Save plot
        comparison_plot_path = './plots/cascade_tradeoff_comparison.png'
        plt.savefig(comparison_plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved cascade comparison plot to {comparison_plot_path}")
    else:
        print("Warning: Aligned Cascade Router not available, skipping gating comparisons.")

if __name__ == "__main__":
    main()
