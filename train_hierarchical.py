"""
train_hierarchical.py

Three-stage training for HierarchicalVQVAE:

  Stage 1  VQ-VAE  — joint training of encoder + both quantizers + decoder
  Stage 2  TopPriorGRU — autoregressive prior over top codebook indices
  Stage 3  BotPriorGRU — autoregressive prior over bottom indices, conditioned on top

Evaluation after each stage reports:
  - Per-prime reconstruction accuracy (VQ-VAE)
  - Top / bottom prior index accuracy
  - Qualitative sample inspection: does the prior generate coherent tree paths?

Usage:
  python train_hierarchical.py [--primes 2 3 5 7 11] [--N 64] ...
"""

import argparse
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from dataset import PadicDataset
from hierarchical_vqvae import HierarchicalVQVAE, TopPriorGRU, BotPriorGRU


# ── Stage 1: VQ-VAE ───────────────────────────────────────────────────────────

def train_vqvae(model, train_loader, val_loader, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='none')

    print('\n--- Stage 1: Training Hierarchical VQ-VAE ---')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        tot_loss = tot_recon = tot_vq = tot_correct = tot_tokens = 0

        for batch in train_loader:
            digits = batch['digits'].to(device)
            p      = batch['p'].to(device)
            optimizer.zero_grad()

            logits, vq_loss, _, _ = model(digits, p)

            B, N, C = logits.shape
            recon_flat   = criterion(logits.reshape(-1, C), digits.reshape(-1))
            recon_sample = recon_flat.reshape(B, N).mean(dim=-1)
            weights      = torch.tensor([math.log(v.item()) + 1.0 for v in p], device=device)
            recon_loss   = (recon_sample * weights).mean()

            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()

            tot_loss    += loss.item()    * B
            tot_recon   += recon_loss.item() * B
            tot_vq      += vq_loss.item() * B
            preds = torch.argmax(logits, dim=-1)
            tot_correct += (preds == digits).sum().item()
            tot_tokens  += B * N

        n  = len(train_loader.dataset)
        ta = tot_correct / tot_tokens

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for batch in val_loader:
                digits = batch['digits'].to(device)
                p      = batch['p'].to(device)
                logits, _, _, _ = model(digits, p)
                preds  = torch.argmax(logits, dim=-1)
                vc    += (preds == digits).sum().item()
                vt    += digits.shape[0] * digits.shape[1]
        va = vc / vt

        print(f'Epoch {epoch+1:02d}/{epochs:02d} | Loss: {tot_loss/n:.4f} '
              f'(Recon: {tot_recon/n:.4f}, VQ: {tot_vq/n:.4f}) | '
              f'Train Acc: {ta*100:.2f}% | Val Acc: {va*100:.2f}%')

    return model


# ── Stage 2: Top Prior ────────────────────────────────────────────────────────

def train_top_prior(vqvae, top_prior, full_loader, epochs, lr, device):
    vqvae.eval()
    print('\nEncoding dataset to get top/bottom indices...')
    all_idx_top, all_idx_bot, all_p = [], [], []

    with torch.no_grad():
        for batch in full_loader:
            digits = batch['digits'].to(device)
            p      = batch['p'].to(device)
            _, _, idx_bot, idx_top, _, _ = vqvae.encode(digits, p)
            all_idx_top.append(idx_top.cpu())
            all_idx_bot.append(idx_bot.cpu())
            all_p.append(p.cpu())

    idx_top = torch.cat(all_idx_top)
    idx_bot = torch.cat(all_idx_bot)
    all_p   = torch.cat(all_p)

    ds  = TensorDataset(idx_top, idx_bot, all_p)
    ldr = DataLoader(ds, batch_size=full_loader.batch_size, shuffle=True)

    print('\n--- Stage 2: Training Top Prior ---')
    top_prior.to(device)
    optimizer = optim.Adam(top_prior.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        top_prior.train()
        tot_loss = tot_correct = tot_tokens = 0

        for it, ib, p in ldr:
            it, ib, p = it.to(device), ib.to(device), p.to(device)
            optimizer.zero_grad()
            logits = top_prior(it, p)
            B, L, K = logits.shape
            loss = criterion(logits.reshape(-1, K), it.reshape(-1))
            loss.backward()
            optimizer.step()

            tot_loss    += loss.item() * B
            preds = torch.argmax(logits, dim=-1)
            tot_correct += (preds == it).sum().item()
            tot_tokens  += B * L

        n = len(ds)
        print(f'Epoch {epoch+1:02d}/{epochs:02d} | Loss: {tot_loss/n:.4f} | '
              f'Top-Prior Acc: {tot_correct/tot_tokens*100:.2f}%')

    return top_prior, idx_top, idx_bot, all_p


# ── Stage 3: Bottom Prior ─────────────────────────────────────────────────────

def train_bot_prior(bot_prior, idx_bot, idx_top, all_p, batch_size, epochs, lr, device):
    ds  = TensorDataset(idx_bot, idx_top, all_p)
    ldr = DataLoader(ds, batch_size=batch_size, shuffle=True)

    print('\n--- Stage 3: Training Bottom Prior ---')
    bot_prior.to(device)
    optimizer = optim.Adam(bot_prior.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        bot_prior.train()
        tot_loss = tot_correct = tot_tokens = 0

        for ib, it, p in ldr:
            ib, it, p = ib.to(device), it.to(device), p.to(device)
            optimizer.zero_grad()
            logits = bot_prior(ib, it, p)
            B, L, K = logits.shape
            loss = criterion(logits.reshape(-1, K), ib.reshape(-1))
            loss.backward()
            optimizer.step()

            tot_loss    += loss.item() * B
            preds = torch.argmax(logits, dim=-1)
            tot_correct += (preds == ib).sum().item()
            tot_tokens  += B * L

        n = len(ds)
        print(f'Epoch {epoch+1:02d}/{epochs:02d} | Loss: {tot_loss/n:.4f} | '
              f'Bot-Prior Acc: {tot_correct/tot_tokens*100:.2f}%')

    return bot_prior


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_per_prime(vqvae, top_prior, bot_prior, primes, N, device, n_samples=5):
    """
    Report per-prime reconstruction accuracy and print a few prior-sampled sequences.
    """
    vqvae.eval(); top_prior.eval(); bot_prior.eval()
    vocab = max(primes) + 2

    print('\n--- Evaluation: Per-Prime Reconstruction & Prior Samples ---')
    for p_val in primes:
        ds = PadicDataset(primes=[p_val], N=N, num_samples_per_type=100)
        ldr = DataLoader(ds, batch_size=128, shuffle=False)

        correct = tokens = 0
        for batch in ldr:
            digits = batch['digits'].to(device)
            p      = batch['p'].to(device)
            logits, _, _, _ = vqvae(digits, p)
            preds  = torch.argmax(logits, dim=-1)
            correct += (preds == digits).sum().item()
            tokens  += digits.shape[0] * digits.shape[1]
        acc = correct / tokens

        # sample from priors
        p_t  = torch.full((n_samples,), p_val, dtype=torch.long, device=device)
        L_top = N // 4
        idx_top_s = top_prior.sample(p_t, L=L_top, temperature=0.8)
        idx_bot_s = bot_prior.sample(idx_top_s, p_t, temperature=0.8)

        # decode
        z_q_top = vqvae.top_quantizer.embedding(idx_top_s)
        z_q_bot = vqvae.bot_quantizer.embedding(idx_bot_s)
        logits  = vqvae.decode(z_q_bot, z_q_top, p_t)
        samples = torch.argmax(logits, dim=-1).cpu().numpy()

        print(f'\n  p={p_val} | Recon acc: {acc*100:.2f}%')
        for i, s in enumerate(samples):
            print(f'    sample {i+1}: {" ".join(str(d) for d in s[:20])} ...')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primes',           type=int, nargs='+', default=[2, 3, 5, 7, 11])
    parser.add_argument('--N',                type=int, default=64)
    parser.add_argument('--samples_per_type', type=int, default=600)
    parser.add_argument('--batch_size',       type=int, default=128)
    parser.add_argument('--vqvae_epochs',     type=int, default=15)
    parser.add_argument('--top_epochs',       type=int, default=12)
    parser.add_argument('--bot_epochs',       type=int, default=12)
    parser.add_argument('--lr',               type=float, default=1e-3)
    parser.add_argument('--hidden_dim',       type=int, default=64)
    parser.add_argument('--bot_codebook',     type=int, default=64)
    parser.add_argument('--top_codebook',     type=int, default=16)
    parser.add_argument('--bot_dim',          type=int, default=32)
    parser.add_argument('--top_dim',          type=int, default=32)
    parser.add_argument('--save_dir',         type=str, default='./checkpoints/hierarchical')
    args = parser.parse_args()

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab     = max(args.primes) + 2
    os.makedirs(args.save_dir, exist_ok=True)

    print(f'Device    : {device}')
    print(f'Primes    : {args.primes}')
    print(f'N         : {args.N}')
    print(f'L_top     : {args.N // 4}  (top latent length)')
    print(f'L_bot     : {args.N // 2}  (bottom latent length)')
    print(f'Codebooks : top={args.top_codebook}, bot={args.bot_codebook}')

    # Dataset
    dataset  = PadicDataset(primes=args.primes, N=args.N,
                            num_samples_per_type=args.samples_per_type)
    val_sz   = int(0.1 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_sz, val_sz])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    full_loader  = DataLoader(dataset,  batch_size=args.batch_size, shuffle=False)

    # ── Stage 1 ───────────────────────────────────────────────────────────
    vqvae = HierarchicalVQVAE(
        vocab_size   = vocab,
        hidden_dim   = args.hidden_dim,
        N            = args.N,
        bot_codebook = args.bot_codebook,
        top_codebook = args.top_codebook,
        bot_dim      = args.bot_dim,
        top_dim      = args.top_dim,
    )
    n_params = sum(p.numel() for p in vqvae.parameters())
    print(f'Parameters: {n_params:,}')

    vqvae = train_vqvae(vqvae, train_loader, val_loader, args.vqvae_epochs, args.lr, device)
    torch.save(vqvae.state_dict(), os.path.join(args.save_dir, 'vqvae.pt'))

    # ── Stage 2 ───────────────────────────────────────────────────────────
    top_prior = TopPriorGRU(
        top_codebook = args.top_codebook,
        top_dim      = args.top_dim,
        hidden_size  = 128,
    )
    top_prior, idx_top, idx_bot, all_p = train_top_prior(
        vqvae, top_prior, full_loader, args.top_epochs, args.lr, device)
    torch.save(top_prior.state_dict(), os.path.join(args.save_dir, 'top_prior.pt'))

    # ── Stage 3 ───────────────────────────────────────────────────────────
    bot_prior = BotPriorGRU(
        bot_codebook = args.bot_codebook,
        top_codebook = args.top_codebook,
        bot_dim      = args.bot_dim,
        top_dim      = args.top_dim,
        hidden_size  = 256,
    )
    bot_prior = train_bot_prior(
        bot_prior, idx_bot, idx_top, all_p,
        args.batch_size, args.bot_epochs, args.lr, device)
    torch.save(bot_prior.state_dict(), os.path.join(args.save_dir, 'bot_prior.pt'))

    # ── Evaluation ────────────────────────────────────────────────────────
    evaluate_per_prime(vqvae, top_prior, bot_prior, args.primes, args.N, device)

    print(f'\nAll checkpoints saved to {args.save_dir}')
    print('Hierarchical VQ-VAE training complete.')


if __name__ == '__main__':
    main()
