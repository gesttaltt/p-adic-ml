import sys, os; root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.extend([root_dir, os.path.join(root_dir, 'src')]); os.chdir(root_dir)
from visualize import plot_padic_tree


def generate_tree_plot(vqvae_path, prior_path, p, vocab_size, N=64,
                       save_path=None, device='cpu', hidden_dim=64):
    """Thin wrapper around visualize.plot_padic_tree for backward compatibility."""
    plot_padic_tree(
        vqvae_path=vqvae_path,
        prior_path=prior_path,
        p=p,
        save_path=save_path,
        save_dir='./plots',
        N=N,
        device=device,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
    )


def main():
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    generate_tree_plot(
        vqvae_path='./checkpoints/broad_p13/vqvae.pt',
        prior_path='./checkpoints/broad_p13/prior.pt',
        p=13, vocab_size=13, N=64, device=device,
        save_path='./plots/padic_tree_13.png',
    )
    generate_tree_plot(
        vqvae_path='./checkpoints/broad_p17/vqvae.pt',
        prior_path='./checkpoints/broad_p17/prior.pt',
        p=17, vocab_size=17, N=64, device=device,
        save_path='./plots/padic_tree_17.png',
    )
    print('Tree visualization plots generated successfully!')


if __name__ == '__main__':
    main()
