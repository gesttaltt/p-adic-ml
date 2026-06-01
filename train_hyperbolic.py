"""
train_hyperbolic.py

Training script for HyperbolicBetaVAE.

Loss = L_recon + β * ||μ_tangent||²

  L_recon : cross-entropy reconstruction (same as ConditionalBetaVAE)
  β term  : pulls the mean tangent vector toward the origin of the ball
             (acts as a regularizer analogous to the Euclidean KL in a standard VAE)

No separate metric-alignment loss is needed: the Poincaré ball's intrinsic
negative curvature embeds tree-structured data with lower distortion than
Euclidean space, so the geometry itself aligns latent distances with
p-adic distances.

Usage:
  python train_hyperbolic.py [--primes 2 3 5 7 11] [--N 64] ...
"""

import argparse
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import PadicDataset
from hyperbolic_vae import HyperbolicBetaVAE
from metric_alignment import compute_hyperbolic_metric_loss


import geoopt.optim

def train(model, train_loader, val_loader, epochs, lr, beta, gamma, device):
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='none')

    print(f"\n--- Training HyperbolicBetaVAE (beta={beta}, gamma={gamma}) ---")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = total_recon = total_reg = total_metric = 0.0
        total_correct = total_tokens = 0

        for batch in train_loader:
            digits = batch['digits'].to(device)
            p      = batch['p'].to(device)

            optimizer.zero_grad()
            logits, mu_tangent, logvar, z_ball = model(digits, p)

            # 1. Reconstruction loss
            B, N, C = logits.shape
            recon_flat   = criterion(logits.reshape(-1, C), digits.reshape(-1))
            recon_sample = recon_flat.reshape(B, N).mean(dim=-1)
            weights      = torch.tensor([math.log(v.item()) + 1.0 for v in p], device=device)
            recon_loss   = (recon_sample * weights).mean()

            # 2. Regularization: pull μ_tangent toward ball origin
            reg_loss = (mu_tangent ** 2).mean()

            # 3. Hyperbolic metric alignment — reuses z_ball from forward (no second sample)
            metric_loss = (
                compute_hyperbolic_metric_loss(z_ball, digits, p, model.manifold)
                if gamma > 0 else torch.tensor(0.0, device=device)
            )

            loss = recon_loss + beta * reg_loss + gamma * metric_loss
            loss.backward()
            optimizer.step()

            total_loss   += loss.item()   * B
            total_recon  += recon_loss.item() * B
            total_reg    += reg_loss.item()   * B
            total_metric += metric_loss.item() * B

            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == digits).sum().item()
            total_tokens  += B * N

        n  = len(train_loader.dataset)
        ta = total_correct / total_tokens

        # Validation accuracy
        model.eval()
        val_correct = val_tokens = 0
        with torch.no_grad():
            for batch in val_loader:
                digits = batch['digits'].to(device)
                p      = batch['p'].to(device)
                logits, _, _, _ = model(digits, p)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == digits).sum().item()
                val_tokens  += digits.shape[0] * digits.shape[1]
        va = val_correct / val_tokens

        # Retrieve current curvature value
        if hasattr(model.manifold, 'c'):
            curv_val = model.manifold.c
            if isinstance(curv_val, torch.Tensor):
                curv_val = curv_val.item()
        elif hasattr(model.manifold, 'k'):
            curv_val = model.manifold.k
            if isinstance(curv_val, torch.Tensor):
                curv_val = curv_val.item()
        else:
            curv_val = 0.0

        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | "
            f"Loss: {total_loss/n:.4f} "
            f"(Recon: {total_recon/n:.4f}, Reg: {total_reg/n:.4f}, Metric: {total_metric/n:.4f}) | "
            f"Curvature: {curv_val:.4f} | "
            f"Train Acc: {ta*100:.2f}% | Val Acc: {va*100:.2f}%"
        )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primes',          type=int,   nargs='+', default=[2, 3, 5, 7, 11])
    parser.add_argument('--N',               type=int,   default=64)
    parser.add_argument('--samples_per_type',type=int,   default=600)
    parser.add_argument('--batch_size',      type=int,   default=128)
    parser.add_argument('--epochs',          type=int,   default=15)
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--beta',            type=float, default=0.05,
                        help='Weight on the origin-pull regularizer')
    parser.add_argument('--gamma',           type=float, default=5.0,
                        help='Weight on hyperbolic metric alignment (0 to disable)')
    parser.add_argument('--curvature',       type=float, default=1.0,
                        help='Initial curvature (c for Poincaré, k for Lorentz)')
    parser.add_argument('--learnable_curvature', action='store_true',
                        help='Optimize the curvature parameter during training')
    parser.add_argument('--manifold',        type=str,   choices=['poincare', 'lorentz'], default='poincare',
                        help='Manifold type to use')
    parser.add_argument('--hidden_dim',      type=int,   default=64)
    parser.add_argument('--latent_dim',      type=int,   default=32)
    parser.add_argument('--save_dir',        type=str,   default='./checkpoints/hyperbolic')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device    : {device}")
    print(f"Primes    : {args.primes}")
    print(f"Curvature : {args.curvature}")

    vocab_size = max(args.primes) + 2

    dataset  = PadicDataset(primes=args.primes, N=args.N, num_samples_per_type=args.samples_per_type)
    val_size = int(0.1 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    model = HyperbolicBetaVAE(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        N=args.N,
        curvature=args.curvature,
        learnable_curvature=args.learnable_curvature,
        manifold=args.manifold,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")

    model = train(model, train_loader, val_loader, args.epochs, args.lr,
                  args.beta, args.gamma, device)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'hyperbolic_vae.pt')
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved to {save_path}")

    # Final evaluation for report
    model.eval()
    val_correct = 0
    val_tokens = 0
    val_metric_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            digits = batch['digits'].to(device)
            p      = batch['p'].to(device)
            logits, mu_tangent, logvar, z_ball = model(digits, p)
            preds = torch.argmax(logits, dim=-1)
            val_correct += (preds == digits).sum().item()
            val_tokens  += digits.shape[0] * digits.shape[1]
            
            # Compute metric loss
            metric_loss = compute_hyperbolic_metric_loss(z_ball, digits, p, model.manifold)
            val_metric_loss += metric_loss.item()
            val_batches += 1
            
    val_acc = val_correct / val_tokens
    avg_val_metric = val_metric_loss / val_batches if val_batches > 0 else 0.0

    # Retrieve final curvature value
    if hasattr(model.manifold, 'c'):
        final_curv = model.manifold.c
        if isinstance(final_curv, torch.Tensor):
            final_curv = final_curv.item()
    elif hasattr(model.manifold, 'k'):
        final_curv = model.manifold.k
        if isinstance(final_curv, torch.Tensor):
            final_curv = final_curv.item()
    else:
        final_curv = args.curvature

    # Save markdown report
    report_path = os.path.join(args.save_dir, 'training_report.md')
    with open(report_path, 'w') as f:
        f.write("# Hyperbolic VAE Training Baseline Report\n\n")
        f.write(f"This baseline configures a large-capacity Hyperbolic/Lorentz Beta-VAE model with learnable curvature.\n\n")
        f.write("## Hyperparameters\n")
        f.write(f"- **Manifold Type**: `{args.manifold}`\n")
        f.write(f"- **Hidden Dimension**: `{args.hidden_dim}`\n")
        f.write(f"- **Latent Dimension**: `{args.latent_dim}`\n")
        f.write(f"- **Sequence Length N**: `{args.N}`\n")
        f.write(f"- **Primes**: `{args.primes}`\n")
        f.write(f"- **Epochs**: `{args.epochs}`\n")
        f.write(f"- **Beta (Reg Cost)**: `{args.beta}`\n")
        f.write(f"- **Gamma (Metric Alignment Cost)**: `{args.gamma}`\n")
        f.write(f"- **Optimized Curvature**: `{final_curv:.5f}`\n\n")
        f.write("## Evaluation Metrics\n")
        f.write(f"- **Final Validation Accuracy**: `{val_acc * 100:.2f}%`\n")
        f.write(f"- **Final Metric Alignment Loss**: `{avg_val_metric:.6f}`\n")
    print(f"Report saved to {report_path}")


if __name__ == '__main__':
    main()
