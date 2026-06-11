import sys, os; _r = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(_r, 'src')); os.chdir(_r)
import unittest
import torch
import numpy as np
import geoopt

from padic_math import mod_inverse, rational_to_padic, padic_to_float, solve_poly_padic
from dataset import PadicDataset
from metric_alignment import batch_padic_distance, compute_metric_loss, compute_hyperbolic_metric_loss
from models import PrimeEmbedder, VectorQuantizer, ConditionalVQVAE, PriorGRU
from beta_vae import ConditionalBetaVAE
from hyperbolic_vae import HyperbolicBetaVAE
from hierarchical_vqvae import HierarchicalVQVAE
from hierarchical_3level import ThreeLevelVQVAE


class TestPadicMath(unittest.TestCase):
    def test_mod_inverse(self):
        self.assertEqual(mod_inverse(3, 5), 2)
        self.assertEqual(mod_inverse(7, 11), 8)
        with self.assertRaises(ValueError):
            mod_inverse(5, 5)

    def test_rational_to_padic(self):
        # -1/3 in base 5
        digits = rational_to_padic(-1, 3, 5, 10)
        self.assertEqual(digits, [3, 1, 3, 1, 3, 1, 3, 1, 3, 1])

        # -1/3 in base 2
        digits2 = rational_to_padic(-1, 3, 2, 10)
        self.assertEqual(digits2, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    def test_solve_poly_padic(self):
        # Solve x^2 + 1 = 0 mod 5^10
        roots = solve_poly_padic([1, 0, 1], 5, 10)
        self.assertEqual(len(roots), 2)
        for r in roots:
            val = padic_to_float(r, 5)
            self.assertEqual(int(val**2 + 1) % (5**10), 0)


class TestDataset(unittest.TestCase):
    def test_padic_dataset(self):
        primes = [2, 3, 5]
        N = 16
        samples_per_type = 10
        ds = PadicDataset(primes=primes, N=N, num_samples_per_type=samples_per_type)
        
        # 3 primes * 3 types (rational, algebraic, random) * 10 samples
        expected_len = len(primes) * 3 * samples_per_type
        self.assertEqual(len(ds), expected_len)

        sample = ds[0]
        self.assertIn('digits', sample)
        self.assertIn('p', sample)
        self.assertIn('type', sample)
        self.assertEqual(sample['digits'].shape, (N,))
        self.assertIn(sample['p'], primes)


class TestMetricAlignment(unittest.TestCase):
    def test_batch_padic_distance(self):
        digits = torch.tensor([
            [1, 0, 1, 0],  # A
            [1, 0, 1, 1],  # B (differs at index 3 from A)
            [1, 1, 0, 0],  # C (differs at index 1 from A)
            [1, 0, 1, 0]   # D (same as A)
        ], dtype=torch.long)
        p = torch.tensor([2, 2, 2, 2], dtype=torch.long)

        D = batch_padic_distance(digits, p)
        self.assertAlmostEqual(D[0, 1].item(), 2**-3)
        self.assertAlmostEqual(D[0, 2].item(), 2**-1)
        self.assertAlmostEqual(D[0, 3].item(), 0.0)

    def test_compute_metric_loss(self):
        z = torch.randn(4, 8)
        digits = torch.tensor([
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0]
        ], dtype=torch.long)
        p = torch.tensor([2, 2, 2, 2], dtype=torch.long)

        loss = compute_metric_loss(z, digits, p)
        self.assertTrue(loss.item() >= 0.0)

    def test_compute_hyperbolic_metric_loss(self):
        manifold = geoopt.PoincareBall(c=1.0)
        z = manifold.random(4, 8)
        digits = torch.tensor([
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0]
        ], dtype=torch.long)
        p = torch.tensor([2, 2, 2, 2], dtype=torch.long)

        loss = compute_hyperbolic_metric_loss(z, digits, p, manifold)
        self.assertTrue(loss.item() >= 0.0)


class TestModels(unittest.TestCase):
    def test_prime_embedder(self):
        embedder = PrimeEmbedder(out_dim=16)
        p = torch.tensor([2, 3, 5], dtype=torch.long)
        emb = embedder(p)
        self.assertEqual(emb.shape, (3, 16))

    def test_vector_quantizer(self):
        vq = VectorQuantizer(num_embeddings=16, embedding_dim=8)
        inputs = torch.randn(4, 5, 8)
        loss, quantized, indices = vq(inputs)
        self.assertEqual(quantized.shape, (4, 5, 8))
        self.assertEqual(indices.shape, (4, 5))
        self.assertTrue(loss.item() >= 0.0)

    def test_conditional_vqvae(self):
        vocab_size = 13
        N = 32
        model = ConditionalVQVAE(vocab_size=vocab_size, hidden_dim=32, codebook_size=16, latent_dim=8, N=N, cond_dim=8)
        digits = torch.randint(0, 2, (4, N))
        p = torch.tensor([2, 2, 2, 2], dtype=torch.long)

        logits, vq_loss, indices = model(digits, p)
        self.assertEqual(logits.shape, (4, N, vocab_size))
        self.assertEqual(indices.shape, (4, N // 2))
        self.assertTrue(vq_loss.item() >= 0.0)

    def test_prior_gru(self):
        model = PriorGRU(codebook_size=16, latent_dim=8, cond_dim=8, hidden_size=32)
        indices = torch.randint(0, 16, (4, 10))
        p = torch.tensor([2, 2, 2, 2], dtype=torch.long)

        logits = model(indices, p)
        self.assertEqual(logits.shape, (4, 10, 16))

        samples = model.sample(p, L=10)
        self.assertEqual(samples.shape, (4, 10))

    def test_conditional_betavae(self):
        vocab_size = 13
        N = 32
        model = ConditionalBetaVAE(vocab_size=vocab_size, hidden_dim=32, latent_dim=8, N=N, cond_dim=8)
        digits = torch.randint(0, 2, (4, N))
        p = torch.tensor([2, 2, 2, 2], dtype=torch.long)

        logits, mu, logvar = model(digits, p)
        self.assertEqual(logits.shape, (4, N, vocab_size))
        self.assertEqual(mu.shape, (4, 8))
        self.assertEqual(logvar.shape, (4, 8))

        samples = model.sample(p)
        self.assertEqual(samples.shape, (4, N))

    def test_hyperbolic_betavae_poincare(self):
        vocab_size = 13
        N = 32
        model = HyperbolicBetaVAE(vocab_size=vocab_size, hidden_dim=32, latent_dim=8, N=N, cond_dim=8,
                                  curvature=1.0, learnable_curvature=True, manifold='poincare')
        digits = torch.randint(0, 2, (4, N))
        p = torch.tensor([2, 2, 2, 2], dtype=torch.long)

        logits, mu_tangent, logvar, z_ball = model(digits, p)
        self.assertEqual(logits.shape, (4, N, vocab_size))
        self.assertEqual(mu_tangent.shape, (4, 8))
        self.assertEqual(logvar.shape, (4, 8))
        self.assertEqual(z_ball.shape, (4, 8))
        self.assertTrue(model.manifold.check_point(z_ball))

        samples = model.sample(p)
        self.assertEqual(samples.shape, (4, N))

    def test_hyperbolic_betavae_lorentz(self):
        vocab_size = 13
        N = 32
        model = HyperbolicBetaVAE(vocab_size=vocab_size, hidden_dim=32, latent_dim=8, N=N, cond_dim=8,
                                  curvature=1.0, learnable_curvature=True, manifold='lorentz')
        digits = torch.randint(0, 2, (4, N))
        p = torch.tensor([2, 2, 2, 2], dtype=torch.long)

        logits, mu_tangent, logvar, z_ball = model(digits, p)
        self.assertEqual(logits.shape, (4, N, vocab_size))
        self.assertEqual(mu_tangent.shape, (4, 8))
        self.assertEqual(logvar.shape, (4, 8))
        self.assertEqual(z_ball.shape, (4, 8))
        self.assertTrue(model.manifold.check_point(z_ball))

        samples = model.sample(p)
        self.assertEqual(samples.shape, (4, N))

    def test_hierarchical_vqvae_attention(self):
        vocab_size = 13
        N = 32
        # Test default (conv decoder)
        model_conv = HierarchicalVQVAE(vocab_size=vocab_size, hidden_dim=32, N=N,
                                       bot_dim=16, top_dim=16, cond_dim=8,
                                       use_attention_decoder=False)
        # Test attention decoder
        model_attn = HierarchicalVQVAE(vocab_size=vocab_size, hidden_dim=32, N=N,
                                       bot_dim=16, top_dim=16, cond_dim=8,
                                       use_attention_decoder=True)
        
        digits = torch.randint(0, 2, (4, N))
        p = torch.tensor([2, 2, 2, 2], dtype=torch.long)
        
        logits_conv, loss_conv, idx_bot_conv, idx_top_conv = model_conv(digits, p)
        logits_attn, loss_attn, idx_bot_attn, idx_top_attn = model_attn(digits, p)
        
        self.assertEqual(logits_conv.shape, (4, N, vocab_size))
        self.assertEqual(logits_attn.shape, (4, N, vocab_size))
        self.assertEqual(idx_bot_conv.shape, (4, N // 2))
        self.assertEqual(idx_bot_attn.shape, (4, N // 2))
        self.assertEqual(idx_top_conv.shape, (4, N // 4))
        self.assertEqual(idx_top_attn.shape, (4, N // 4))
        self.assertTrue(loss_conv.item() >= 0.0)
        self.assertTrue(loss_attn.item() >= 0.0)

    def test_three_level_vqvae_attention(self):
        vocab_size = 13
        N = 32
        # Test default (conv decoder)
        model_conv = ThreeLevelVQVAE(vocab_size=vocab_size, hidden_dim=32, N=N,
                                     bot_dim=16, mid_dim=16, top_dim=16, cond_dim=8,
                                     use_attention_decoder=False)
        # Test attention decoder
        model_attn = ThreeLevelVQVAE(vocab_size=vocab_size, hidden_dim=32, N=N,
                                     bot_dim=16, mid_dim=16, top_dim=16, cond_dim=8,
                                     use_attention_decoder=True)
        
        digits = torch.randint(0, 2, (4, N))
        p = torch.tensor([2, 2, 2, 2], dtype=torch.long)
        
        logits_conv, loss_conv, idx_bot_conv, idx_mid_conv, idx_top_conv = model_conv(digits, p)
        logits_attn, loss_attn, idx_bot_attn, idx_mid_attn, idx_top_attn = model_attn(digits, p)
        
        self.assertEqual(logits_conv.shape, (4, N, vocab_size))
        self.assertEqual(logits_attn.shape, (4, N, vocab_size))
        self.assertEqual(idx_bot_conv.shape, (4, N // 2))
        self.assertEqual(idx_bot_attn.shape, (4, N // 2))
        self.assertEqual(idx_mid_conv.shape, (4, N // 4))
        self.assertEqual(idx_mid_attn.shape, (4, N // 4))
        self.assertEqual(idx_top_conv.shape, (4, N // 8))
        self.assertEqual(idx_top_attn.shape, (4, N // 8))
        self.assertTrue(loss_conv.item() >= 0.0)
        self.assertTrue(loss_attn.item() >= 0.0)


if __name__ == '__main__':
    unittest.main()
