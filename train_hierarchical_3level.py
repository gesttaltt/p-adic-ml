"""
train_hierarchical_3level.py  —  Item #25

Three-stage training for ThreeLevelVQVAE on N=128 sequences:
  Stage 1  VQ-VAE     — joint training of all 3 levels + decoder
  Stage 2  TopPrior   — autoregressive over N/8 top tokens
  Stage 3  MidPrior   — autoregressive over N/4 mid tokens | top
  Stage 4  BotPrior   — autoregressive over N/2 bot tokens | mid, top

Usage:
  python train_hierarchical_3level.py [--primes 2 3 5 7 11] [--N 128] ...
"""

import argparse, math, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from dataset import PadicDataset
from hierarchical_3level import (ThreeLevelVQVAE,
                                  ThreeLevelTopPriorGRU,
                                  ThreeLevelMidPriorGRU,
                                  ThreeLevelBotPriorGRU)


# ── Stage 1: VQ-VAE ────────────────────────────────────────────────────────────

def train_vqvae(model, train_loader, val_loader, epochs, lr, device):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(reduction='none')
    print('\n--- Stage 1: Training Three-Level VQ-VAE ---')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        tot_loss = tot_recon = tot_vq = tot_correct = tot_tokens = 0
        for batch in train_loader:
            digits = batch['digits'].to(device); p = batch['p'].to(device)
            opt.zero_grad()
            logits, vq_loss, _, _, _ = model(digits, p)
            B, N, C = logits.shape
            recon_flat = crit(logits.reshape(-1, C), digits.reshape(-1))
            recon = (recon_flat.reshape(B, N).mean(-1) *
                     torch.tensor([math.log(v.item())+1 for v in p], device=device)).mean()
            loss = recon + vq_loss
            loss.backward(); opt.step()
            tot_loss += loss.item()*B; tot_recon += recon.item()*B; tot_vq += vq_loss.item()*B
            tot_correct += (torch.argmax(logits,-1)==digits).sum().item(); tot_tokens += B*N

        n = len(train_loader.dataset); ta = tot_correct/tot_tokens
        model.eval(); vc=vt=0
        with torch.no_grad():
            for batch in val_loader:
                digits=batch['digits'].to(device); p=batch['p'].to(device)
                logits,_,_,_,_=model(digits,p)
                vc+=(torch.argmax(logits,-1)==digits).sum().item(); vt+=digits.shape[0]*digits.shape[1]
        print(f'Epoch {epoch+1:02d}/{epochs} | Loss {tot_loss/n:.4f} '
              f'(Recon {tot_recon/n:.4f} VQ {tot_vq/n:.4f}) | '
              f'Train {ta*100:.2f}% | Val {vc/vt*100:.2f}%')
    return model


# ── Stage 2: Top Prior ─────────────────────────────────────────────────────────

def encode_all(model, loader, device):
    model.eval(); idxs_top=[]; idxs_mid=[]; idxs_bot=[]; primes=[]
    with torch.no_grad():
        for batch in loader:
            d=batch['digits'].to(device); p=batch['p'].to(device)
            _,_,_,ib,im,it,_ = model.encode(d, p)
            idxs_bot.append(ib.cpu()); idxs_mid.append(im.cpu())
            idxs_top.append(it.cpu()); primes.append(p.cpu())
    return (torch.cat(idxs_bot), torch.cat(idxs_mid),
            torch.cat(idxs_top), torch.cat(primes))


def train_prior(prior, ds, batch_size, epochs, lr, device, label):
    ldr = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = optim.Adam(prior.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    print(f'\n--- Stage: Training {label} ---')
    prior.to(device)

    for epoch in range(epochs):
        prior.train(); tot_loss=tot_correct=tot_tokens=0
        for batch in ldr:
            items = [b.to(device) for b in batch]
            opt.zero_grad()
            # first item is always the target indices
            if len(items) == 2:          # (idx_target, p)
                logits = prior(items[0], items[1])
                target = items[0]
            elif len(items) == 3:        # (idx_target, idx_cond1, p)
                logits = prior(items[0], items[1], items[2])
                target = items[0]
            else:                        # (idx_bot, idx_mid, idx_top, p)
                logits = prior(items[0], items[1], items[2], items[3])
                target = items[0]
            B,L,K = logits.shape
            loss = crit(logits.reshape(-1,K), target.reshape(-1))
            loss.backward(); opt.step()
            tot_loss += loss.item()*B
            tot_correct += (torch.argmax(logits,-1)==target).sum().item()
            tot_tokens += B*L
        print(f'Epoch {epoch+1:02d}/{epochs} | Loss {tot_loss/len(ds):.4f} | Acc {tot_correct/tot_tokens*100:.2f}%')
    return prior


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, top_prior, mid_prior, bot_prior, primes, N, device, n_samples=5):
    model.eval(); top_prior.eval(); mid_prior.eval(); bot_prior.eval()
    print('\n--- Evaluation: Per-Prime Reconstruction & Samples ---')
    for p_val in primes:
        ds = PadicDataset(primes=[p_val], N=N, num_samples_per_type=100)
        ldr = DataLoader(ds, batch_size=128)
        correct=tokens=0
        for batch in ldr:
            d=batch['digits'].to(device); p=batch['p'].to(device)
            logits,_,_,_,_ = model(d,p)
            correct+=(torch.argmax(logits,-1)==d).sum().item(); tokens+=d.shape[0]*d.shape[1]

        p_t = torch.full((n_samples,), p_val, dtype=torch.long, device=device)
        it  = top_prior.sample(p_t, L=N//8, temperature=0.8)
        im  = mid_prior.sample(it, p_t, temperature=0.8)
        ib  = bot_prior.sample(im, it, p_t, temperature=0.8)
        z_top = model.top_quantizer.embedding(it)
        z_mid = model.mid_quantizer.embedding(im)
        z_bot = model.bot_quantizer.embedding(ib)
        seqs  = torch.argmax(model.decode(z_bot, z_mid, z_top, p_t), dim=-1).cpu().numpy()

        print(f'\n  p={p_val} | Recon {correct/tokens*100:.2f}%')
        for i, s in enumerate(seqs):
            print(f'    sample {i+1}: {" ".join(str(d) for d in s[:24])} ...')


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primes',           type=int,   nargs='+', default=[2,3,5,7,11])
    parser.add_argument('--N',                type=int,   default=128)
    parser.add_argument('--samples_per_type', type=int,   default=400)
    parser.add_argument('--batch_size',       type=int,   default=64)
    parser.add_argument('--vqvae_epochs',     type=int,   default=15)
    parser.add_argument('--prior_epochs',     type=int,   default=12)
    parser.add_argument('--lr',               type=float, default=1e-3)
    parser.add_argument('--hidden_dim',       type=int,   default=64)
    parser.add_argument('--bot_codebook',     type=int,   default=64)
    parser.add_argument('--mid_codebook',     type=int,   default=32)
    parser.add_argument('--top_codebook',     type=int,   default=16)
    parser.add_argument('--save_dir',         type=str,   default='./checkpoints/hierarchical_3level')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab  = max(args.primes) + 2
    os.makedirs(args.save_dir, exist_ok=True)

    print(f'Device : {device}')
    print(f'Primes : {args.primes}  N={args.N}')
    print(f'Levels : top={args.N//8} mid={args.N//4} bot={args.N//2} tokens')

    ds       = PadicDataset(primes=args.primes, N=args.N, num_samples_per_type=args.samples_per_type)
    val_sz   = int(0.1*len(ds))
    tr_ds, val_ds = random_split(ds, [len(ds)-val_sz, val_sz])
    train_loader  = DataLoader(tr_ds,  batch_size=args.batch_size, shuffle=True)
    val_loader    = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    full_loader   = DataLoader(ds,     batch_size=args.batch_size, shuffle=False)

    # Stage 1
    model = ThreeLevelVQVAE(vocab_size=vocab, hidden_dim=args.hidden_dim, N=args.N,
                             bot_codebook=args.bot_codebook, mid_codebook=args.mid_codebook,
                             top_codebook=args.top_codebook)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    model = train_vqvae(model, train_loader, val_loader, args.vqvae_epochs, args.lr, device)
    torch.save(model.state_dict(), f'{args.save_dir}/vqvae.pt')

    # Encode all
    print('\nEncoding dataset...')
    idx_bot, idx_mid, idx_top, all_p = encode_all(model, full_loader, device)

    # Stage 2: top prior
    top_prior = ThreeLevelTopPriorGRU(top_codebook=args.top_codebook, top_dim=32)
    top_prior = train_prior(top_prior,
                            TensorDataset(idx_top, all_p),
                            args.batch_size, args.prior_epochs, args.lr, device, 'Top Prior')
    torch.save(top_prior.state_dict(), f'{args.save_dir}/top_prior.pt')

    # Stage 3: mid prior
    mid_prior = ThreeLevelMidPriorGRU(mid_codebook=args.mid_codebook,
                                       top_codebook=args.top_codebook, mid_dim=32, top_dim=32)
    mid_prior = train_prior(mid_prior,
                            TensorDataset(idx_mid, idx_top, all_p),
                            args.batch_size, args.prior_epochs, args.lr, device, 'Mid Prior')
    torch.save(mid_prior.state_dict(), f'{args.save_dir}/mid_prior.pt')

    # Stage 4: bot prior
    bot_prior = ThreeLevelBotPriorGRU(bot_codebook=args.bot_codebook,
                                       mid_codebook=args.mid_codebook,
                                       top_codebook=args.top_codebook,
                                       bot_dim=32, mid_dim=32, top_dim=32)
    bot_prior = train_prior(bot_prior,
                            TensorDataset(idx_bot, idx_mid, idx_top, all_p),
                            args.batch_size, args.prior_epochs, args.lr, device, 'Bot Prior')
    torch.save(bot_prior.state_dict(), f'{args.save_dir}/bot_prior.pt')

    evaluate(model, top_prior, mid_prior, bot_prior, args.primes, args.N, device)
    print(f'\nAll checkpoints saved to {args.save_dir}')


if __name__ == '__main__':
    main()
