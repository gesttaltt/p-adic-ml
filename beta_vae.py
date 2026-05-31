import torch
import torch.nn as nn
import torch.nn.functional as F
from models import PrimeEmbedder

class ConditionalBetaVAE(nn.Module):
    def __init__(self, vocab_size=13, hidden_dim=64, latent_dim=32, N=32, cond_dim=16):
        super(ConditionalBetaVAE, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.N = N
        self.cond_dim = cond_dim

        # Embeddings
        self.digit_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, N, hidden_dim))
        self.prime_emb = PrimeEmbedder(cond_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        
        # Encoder (downsamples N -> N/4)
        self.enc_conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1) # N -> N/2
        self.enc_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1) # N/2 -> N/4
        
        self.flatten_dim = hidden_dim * (N // 4)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder (upsamples N/4 -> N)
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)
        self.dec_deconv1 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1) # N/4 -> N/2
        self.dec_deconv2 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1) # N/2 -> N
        self.dec_proj = nn.Linear(hidden_dim, vocab_size)
        
    def encode(self, digits, p):
        # digits: [B, N], p: [B]
        B = digits.shape[0]
        x = self.digit_emb(digits) + self.pos_emb # [B, N, hidden_dim]
        
        cond = self.cond_proj(self.prime_emb(p)).unsqueeze(1) # [B, 1, hidden_dim]
        x = x + cond
        
        x = x.transpose(1, 2) # [B, hidden_dim, N]
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x)) # [B, hidden_dim, N/4]
        
        x_flat = x.reshape(B, -1)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z, p):
        # z: [B, latent_dim], p: [B]
        B = z.shape[0]
        x = F.relu(self.fc_dec(z)) # [B, flatten_dim]
        x = x.reshape(B, self.hidden_dim, self.N // 4) # [B, hidden_dim, N/4]
        
        x = F.relu(self.dec_deconv1(x)) # [B, hidden_dim, N/2]
        x = F.relu(self.dec_deconv2(x)) # [B, hidden_dim, N]
        x = x.transpose(1, 2) # [B, N, hidden_dim]
        
        cond = self.cond_proj(self.prime_emb(p)).unsqueeze(1)
        x = x + cond
        
        logits = self.dec_proj(x) # [B, N, vocab_size]
        
        # Logit Masking: mask out digits >= p
        mask = torch.arange(self.vocab_size, device=logits.device).unsqueeze(0).unsqueeze(0) >= p.unsqueeze(1).unsqueeze(2)
        logits = logits.masked_fill(mask, -1e9)
        
        return logits
        
    def forward(self, digits, p):
        mu, logvar = self.encode(digits, p)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, p)
        return logits, mu, logvar
        
    @torch.no_grad()
    def sample(self, p, device='cpu'):
        # p: [B]
        B = p.shape[0]
        z = torch.randn(B, self.latent_dim, device=device)
        logits = self.decode(z, p)
        return torch.argmax(logits, dim=-1)
