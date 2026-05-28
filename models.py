import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        
    def forward(self, inputs):
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # Calculate distances: ||x - e||^2 = ||x||^2 + ||e||^2 - 2*x*e
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
                     
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = encoding_indices.reshape(inputs.shape[0], inputs.shape[1])
        
        # Quantize
        quantized = self.embedding(encoding_indices)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return loss, quantized, encoding_indices

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out

class ConditionalVQVAE(nn.Module):
    def __init__(self, vocab_size=13, hidden_dim=64, codebook_size=64, latent_dim=32, N=32, cond_dim=16, prime_vocab_size=20):
        super(ConditionalVQVAE, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.N = N
        self.cond_dim = cond_dim
        self.prime_vocab_size = prime_vocab_size
        
        # Embeddings
        self.digit_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, N, hidden_dim))
        self.prime_emb = nn.Embedding(prime_vocab_size, cond_dim)  # supporting primes up to prime_vocab_size-1
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        
        # Encoder (downsamples N -> N/2)
        self.enc_conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1) # N -> N/2
        self.enc_res1 = ResidualBlock(hidden_dim)
        self.enc_conv2 = nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        
        # Quantizer
        self.quantizer = VectorQuantizer(codebook_size, latent_dim)
        
        # Decoder (upsamples N/2 -> N)
        self.dec_conv1 = nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.dec_res1 = ResidualBlock(hidden_dim)
        self.dec_deconv = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1) # N/2 -> N
        self.dec_proj = nn.Linear(hidden_dim, vocab_size)
        
    def encode(self, digits, p):
        # digits: [B, N], p: [B]
        B, N = digits.shape
        x = self.digit_emb(digits) + self.pos_emb # [B, N, hidden_dim]
        
        cond = self.cond_proj(self.prime_emb(p)).unsqueeze(1) # [B, 1, hidden_dim]
        x = x + cond # broadcast condition
        
        x = x.transpose(1, 2) # [B, hidden_dim, N]
        x = F.relu(self.enc_conv1(x))
        x = self.enc_res1(x)
        x = self.enc_conv2(x) # [B, latent_dim, N/2]
        x = x.transpose(1, 2) # [B, N/2, latent_dim]
        return x
        
    def decode(self, quantized, p):
        # quantized: [B, N/2, latent_dim], p: [B]
        x = quantized.transpose(1, 2) # [B, latent_dim, N/2]
        x = F.relu(self.dec_conv1(x))
        x = self.dec_res1(x)
        x = F.relu(self.dec_deconv(x)) # [B, hidden_dim, N]
        x = x.transpose(1, 2) # [B, N, hidden_dim]
        
        # Inject condition in decoder as well
        cond = self.cond_proj(self.prime_emb(p)).unsqueeze(1)
        x = x + cond
        
        logits = self.dec_proj(x) # [B, N, vocab_size]
        
        # Logit Masking: mask out digits >= p
        # vocab_size is 13.
        mask = torch.arange(self.vocab_size, device=logits.device).unsqueeze(0).unsqueeze(0) >= p.unsqueeze(1).unsqueeze(2)
        logits = logits.masked_fill(mask, -1e9)
        
        return logits
        
    def forward(self, digits, p):
        z_e = self.encode(digits, p)
        vq_loss, z_q, indices = self.quantizer(z_e)
        logits = self.decode(z_q, p)
        return logits, vq_loss, indices

class PriorGRU(nn.Module):
    def __init__(self, codebook_size=64, latent_dim=32, cond_dim=16, hidden_size=128, num_layers=2, prime_vocab_size=20):
        super(PriorGRU, self).__init__()
        self.codebook_size = codebook_size
        # The input vocabulary is codebook_size + 1 (the last index is the SOS token)
        self.vocab_size = codebook_size + 1
        self.sos_token = codebook_size
        self.prime_vocab_size = prime_vocab_size
        
        self.token_emb = nn.Embedding(self.vocab_size, latent_dim)
        self.prime_emb = nn.Embedding(prime_vocab_size, cond_dim)
        
        # Input to GRU: token embedding + prime embedding
        self.gru = nn.GRU(
            input_size=latent_dim + cond_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.proj = nn.Linear(hidden_size, codebook_size)
        
    def forward(self, indices, p):
        # indices shape: [B, L]
        # p shape: [B]
        B, L = indices.shape
        
        # Prepend SOS token
        sos = torch.full((B, 1), self.sos_token, dtype=torch.long, device=indices.device)
        inp_seq = torch.cat([sos, indices[:, :-1]], dim=1) # [B, L]
        
        x_emb = self.token_emb(inp_seq) # [B, L, latent_dim]
        p_emb = self.prime_emb(p).unsqueeze(1).repeat(1, L, 1) # [B, L, cond_dim]
        
        inp = torch.cat([x_emb, p_emb], dim=-1) # [B, L, latent_dim + cond_dim]
        
        out, _ = self.gru(inp) # [B, L, hidden_size]
        logits = self.proj(out) # [B, L, codebook_size]
        
        return logits

    @torch.no_grad()
    def sample(self, p, L=16, temperature=1.0):
        # p shape: [B]
        B = p.shape[0]
        device = p.device
        
        curr_tokens = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)
        all_tokens = []
        
        # We need to maintain GRU state manually for step-by-step sampling
        p_emb = self.prime_emb(p).unsqueeze(1) # [B, 1, cond_dim]
        hidden = None
        
        for t in range(L):
            x_emb = self.token_emb(curr_tokens) # [B, 1, latent_dim]
            inp = torch.cat([x_emb, p_emb], dim=-1) # [B, 1, latent_dim + cond_dim]
            
            out, hidden = self.gru(inp, hidden) # out: [B, 1, hidden_size]
            logits = self.proj(out[:, 0, :]) # [B, codebook_size]
            
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1)
            else:
                curr_tokens = torch.argmax(logits, dim=-1, keepdim=True)
                
            all_tokens.append(curr_tokens)
            
        return torch.cat(all_tokens, dim=1) # [B, L]
