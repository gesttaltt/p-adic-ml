import sys, os; _r = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(_r, 'src')); os.chdir(_r)
import argparse
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import PadicDataset
from hyperbolic_vae import HyperbolicBetaVAE
from metric_alignment import compute_hyperbolic_metric_loss
import geoopt.optim


def run_experiment(config, train_loader, val_loader, args, vocab_size, device):
    curvature = config['curvature']
    learnable = config['learnable']
    
    print(f"\n==================================================")
    print(f"Running sweep: curvature={curvature}, learnable={learnable}")
    print(f"==================================================")

    model = HyperbolicBetaVAE(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        N=args.N,
        curvature=curvature,
        learnable_curvature=learnable,
        manifold=args.manifold
    ).to(device)

    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Train loop
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            digits = batch['digits'].to(device)
            p = batch['p'].to(device)

            optimizer.zero_grad()
            logits, mu_tangent, logvar, z_ball = model(digits, p)

            # Recon loss
            B, N, C = logits.shape
            recon_flat = criterion(logits.reshape(-1, C), digits.reshape(-1))
            recon_sample = recon_flat.reshape(B, N).mean(dim=-1)
            weights = torch.tensor([math.log(v.item()) + 1.0 for v in p], device=device)
            recon_loss = (recon_sample * weights).mean()

            # Reg loss
            reg_loss = (mu_tangent ** 2).mean()

            # Metric loss
            metric_loss = compute_hyperbolic_metric_loss(z_ball, digits, p, model.manifold)

            loss = recon_loss + args.beta * reg_loss + args.gamma * metric_loss
            loss.backward()
            optimizer.step()

    # Evaluate on val set
    model.eval()
    val_correct = 0
    val_tokens = 0
    val_metric_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            digits = batch['digits'].to(device)
            p = batch['p'].to(device)
            
            logits, mu_tangent, logvar, z_ball = model(digits, p)
            preds = torch.argmax(logits, dim=-1)
            val_correct += (preds == digits).sum().item()
            val_tokens += digits.shape[0] * digits.shape[1]
            
            # Compute metric loss
            metric_loss = compute_hyperbolic_metric_loss(z_ball, digits, p, model.manifold)
            val_metric_loss += metric_loss.item()
            val_batches += 1

    val_acc = val_correct / val_tokens
    avg_val_metric = val_metric_loss / val_batches if val_batches > 0 else 0.0

    # Read final curvature
    if hasattr(model.manifold, 'c'):
        final_curv = model.manifold.c
        if isinstance(final_curv, torch.Tensor):
            final_curv = final_curv.item()
    elif hasattr(model.manifold, 'k'):
        final_curv = model.manifold.k
        if isinstance(final_curv, torch.Tensor):
            final_curv = final_curv.item()
    else:
        final_curv = curvature

    return {
        'initial_curvature': curvature,
        'learnable': learnable,
        'final_curvature': final_curv,
        'val_accuracy': val_acc,
        'val_metric_alignment_loss': avg_val_metric
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primes',          type=int,   nargs='+', default=[2, 3, 5, 7, 11])
    parser.add_argument('--N',               type=int,   default=64)
    parser.add_argument('--samples_per_type',type=int,   default=200)
    parser.add_argument('--batch_size',      type=int,   default=128)
    parser.add_argument('--epochs',          type=int,   default=5)
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--beta',            type=float, default=0.05)
    parser.add_argument('--gamma',           type=float, default=5.0)
    parser.add_argument('--hidden_dim',      type=int,   default=64)
    parser.add_argument('--latent_dim',      type=int,   default=32)
    parser.add_argument('--manifold',        type=str,   choices=['poincare', 'lorentz'], default='poincare')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sweep running on: {device}")
    print(f"Manifold        : {args.manifold}")
    print(f"Primes          : {args.primes}")
    print(f"Epochs          : {args.epochs}")

    vocab_size = max(args.primes) + 2

    # Share same dataset split to reduce variance and save time
    dataset = PadicDataset(primes=args.primes, N=args.N, num_samples_per_type=args.samples_per_type)
    val_size = int(0.15 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    configs = [
        {'curvature': 0.5, 'learnable': False},
        {'curvature': 1.0, 'learnable': False},
        {'curvature': 2.0, 'learnable': False},
        {'curvature': 5.0, 'learnable': False},
        {'curvature': 1.0, 'learnable': True},
    ]

    results = []
    for config in configs:
        res = run_experiment(config, train_loader, val_loader, args, vocab_size, device)
        results.append(res)

    # Compile report
    report_lines = []
    report_lines.append(f"# Curvature Optimization Sweep ({args.manifold.capitalize()} Manifold)")
    report_lines.append(f"Primes evaluated: {args.primes} | Epochs per run: {args.epochs}\n")
    report_lines.append("| Config (Curvature, Learnable) | Final Curvature | Val Acc (%) | Val Metric Alignment |")
    report_lines.append("| :--- | :---: | :---: | :---: |")

    for r in results:
        cfg_str = f"c={r['initial_curvature']} (Fixed)" if not r['learnable'] else f"c={r['initial_curvature']} (Learnable)"
        acc_str = f"{r['val_accuracy']*100:.2f}%"
        report_lines.append(
            f"| {cfg_str} | {r['final_curvature']:.4f} | {acc_str} | {r['val_metric_alignment_loss']:.5f} |"
        )

    report_content = "\n".join(report_lines)
    print("\n\n" + report_content + "\n\n")

    # Determine saving path
    artifacts_dir = os.environ.get("ARTIFACTS_DIR", "./scaling_analysis")
    os.makedirs(artifacts_dir, exist_ok=True)
    save_path = os.path.join(artifacts_dir, f"curvature_sweep_{args.manifold}.md")
    with open(save_path, "w") as f:
        f.write(report_content)
    print(f"Saved sweep report to {os.path.abspath(save_path)}")


if __name__ == '__main__':
    main()
