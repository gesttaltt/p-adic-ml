import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import PadicDataset
from beta_vae import ConditionalBetaVAE
from metric_alignment import compute_metric_loss

def train_beta_vae_metric(model, train_loader, val_loader, epochs, lr, beta, gamma, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    print(f"\n--- Training Conditional Beta-VAE with Metric Alignment (gamma={gamma}) ---")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_metric = 0
        total_correct = 0
        total_tokens = 0
        
        for batch in train_loader:
            digits = batch['digits'].to(device) # [B, N]
            p = batch['p'].to(device) # [B]
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, mu, logvar = model(digits, p) # logits: [B, N, vocab_size]
            
            # Reconstruct latent z
            z = model.reparameterize(mu, logvar)
            
            # 1. Reconstruction Loss
            B, N, C = logits.shape
            recon_loss = criterion(logits.reshape(-1, C), digits.reshape(-1)).mean()
            
            # 2. KL Loss
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # 3. Metric Alignment Loss
            metric_loss = compute_metric_loss(z, digits, p)
            
            # Total Loss
            loss = recon_loss + beta * kl_loss + gamma * metric_loss
            loss.backward()
            optimizer.step()
            
            # Metrics accumulation
            total_loss += loss.item() * B
            total_recon += recon_loss.item() * B
            total_kl += kl_loss.item() * B
            total_metric += metric_loss.item() * B
            
            # Accuracy
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == digits).sum().item()
            total_tokens += B * N
            
        train_loss = total_loss / len(train_loader.dataset)
        train_recon = total_recon / len(train_loader.dataset)
        train_kl = total_kl / len(train_loader.dataset)
        train_metric = total_metric / len(train_loader.dataset)
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
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}, Metric: {train_metric:.4f}) | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primes', type=int, nargs='+', default=[2, 3, 5, 7])
    parser.add_argument('--N', type=int, default=32)
    parser.add_argument('--samples_per_type', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--beta', type=float, default=1.5, help='KL weight')
    parser.add_argument('--gamma', type=float, default=10.0, help='Metric alignment weight')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset
    dataset = PadicDataset(primes=args.primes, N=args.N, num_samples_per_type=args.samples_per_type)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Model
    model = ConditionalBetaVAE(vocab_size=13, hidden_dim=64, latent_dim=32, N=args.N)
    
    # 3. Train
    model = train_beta_vae_metric(model, train_loader, val_loader, args.epochs, args.lr, args.beta, args.gamma, device)
    
    # 4. Save
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'beta_vae_metric.pt')
    torch.save(model.state_dict(), save_path)
    print(f"Saved metric-aligned VAE checkpoints to {save_path}")

if __name__ == "__main__":
    main()
