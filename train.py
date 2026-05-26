import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from dataset import PadicDataset
from models import ConditionalVQVAE, PriorGRU

def train_vqvae(model, train_loader, val_loader, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='none') # We'll compute loss element-wise to handle masking properly
    
    print("\n--- Stage 1: Training VQ-VAE ---")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_vq = 0
        total_correct = 0
        total_tokens = 0
        
        for batch in train_loader:
            digits = batch['digits'].to(device) # [B, N]
            p = batch['p'].to(device) # [B]
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, vq_loss, _ = model(digits, p) # logits: [B, N, vocab_size]
            
            # Loss calculations
            # Reshape for cross entropy: logits [B, N, C] -> [B*N, C], targets [B, N] -> [B*N]
            B, N, C = logits.shape
            logits_flat = logits.reshape(-1, C)
            targets_flat = digits.reshape(-1)
            
            recon_loss_flat = criterion(logits_flat, targets_flat)
            recon_loss_sample = recon_loss_flat.reshape(B, N).mean(dim=-1)
            import math
            weights = torch.tensor([math.log(val.item()) + 1.0 for val in p], device=device)
            recon_loss = (recon_loss_sample * weights).mean()
            
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item() * B
            total_recon += recon_loss.item() * B
            total_vq += vq_loss.item() * B
            
            # Accuracy
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == digits).sum().item()
            total_tokens += B * N
            
        train_loss = total_loss / len(train_loader.dataset)
        train_recon = total_recon / len(train_loader.dataset)
        train_vq = total_vq / len(train_loader.dataset)
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
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, VQ: {train_vq:.4f}) | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
    return model

def train_prior(vqvae, prior, train_loader, epochs, lr, device):
    vqvae.eval()
    
    # 1. Encode all dataset samples to codebook indices
    print("\nEncoding dataset into discrete latents...")
    all_indices = []
    all_primes = []
    
    with torch.no_grad():
        for batch in train_loader:
            digits = batch['digits'].to(device)
            p = batch['p'].to(device)
            z_e = vqvae.encode(digits, p)
            _, _, indices = vqvae.quantizer(z_e) # [B, L]
            
            all_indices.append(indices.cpu())
            all_primes.append(p.cpu())
            
    all_indices = torch.cat(all_indices, dim=0) # [Dataset_Size, L]
    all_primes = torch.cat(all_primes, dim=0) # [Dataset_Size]
    
    # 2. Create tensor dataset for prior training
    prior_dataset = TensorDataset(all_indices, all_primes)
    prior_loader = DataLoader(prior_dataset, batch_size=train_loader.batch_size, shuffle=True)
    
    # 3. Train prior
    print("\n--- Stage 2: Training Autoregressive Prior ---")
    prior.to(device)
    optimizer = optim.Adam(prior.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        prior.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        for indices, p in prior_loader:
            indices = indices.to(device)
            p = p.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: predict next token in sequence
            logits = prior(indices, p) # [B, L, codebook_size]
            
            # Loss calculations
            B, L, K = logits.shape
            loss = criterion(logits.reshape(-1, K), indices.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * B
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == indices).sum().item()
            total_tokens += B * L
            
        train_loss = total_loss / len(prior_dataset)
        train_acc = total_correct / total_tokens
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {train_loss:.4f} | Prior Index Accuracy: {train_acc*100:.2f}%")
        
    return prior

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primes', type=int, nargs='+', default=[2, 3, 5, 7, 11])
    parser.add_argument('--N', type=int, default=32, help='Length of p-adic expansion')
    parser.add_argument('--samples_per_type', type=int, default=1000, help='Number of samples per class per prime')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--vqvae_epochs', type=int, default=15)
    parser.add_argument('--prior_epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--codebook_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Dataset & Loaders
    dataset = PadicDataset(primes=args.primes, N=args.N, num_samples_per_type=args.samples_per_type)
    
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. VQ-VAE
    vqvae = ConditionalVQVAE(
        vocab_size=13, # up to prime 11/13
        hidden_dim=args.hidden_dim,
        codebook_size=args.codebook_size,
        latent_dim=args.latent_dim,
        N=args.N,
        cond_dim=16
    )
    
    vqvae = train_vqvae(vqvae, train_loader, val_loader, args.vqvae_epochs, args.lr, device)
    torch.save(vqvae.state_dict(), os.path.join(args.save_dir, 'vqvae.pt'))
    print(f"Saved VQ-VAE checkpoints to {args.save_dir}/vqvae.pt")
    
    # 3. Autoregressive Prior
    prior = PriorGRU(
        codebook_size=args.codebook_size,
        latent_dim=args.latent_dim,
        cond_dim=16,
        hidden_size=128,
        num_layers=2
    )
    
    # We pass the full train_loader to build latent dataset (including val elements, or we can just use the train dataset)
    # Let's use a unified dataloader of the whole dataset for encoding, to maximize prior training size
    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    prior = train_prior(vqvae, prior, full_loader, args.prior_epochs, args.lr, device)
    torch.save(prior.state_dict(), os.path.join(args.save_dir, 'prior.pt'))
    print(f"Saved Prior checkpoints to {args.save_dir}/prior.pt")

if __name__ == "__main__":
    main()
