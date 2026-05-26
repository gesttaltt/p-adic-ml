import os
import subprocess
import sys

def main():
    print("======================================================================")
    print("        p-adic VAE Scale-up Pipeline to N=64 digits                   ")
    print("======================================================================")
    
    python_bin = os.path.join("venv", "bin", "python")
    if not os.path.exists(python_bin):
        python_bin = sys.executable
        print(f"Using fallback python: {python_bin}")
    else:
        print(f"Using virtual environment python: {python_bin}")
        
    checkpoints_dir = "./checkpoints"
    
    # Configuration
    N = 64
    vqvae_epochs = 12
    prior_epochs = 12
    vae_epochs = 15
    samples_per_type = 600
    beta = 0.05
    gamma = 5.0
    
    # 1. Train VQ-VAE + Prior (Stage 1 & 2)
    vq_train_cmd = [
        python_bin, "train.py",
        "--N", str(N),
        "--vqvae_epochs", str(vqvae_epochs),
        "--prior_epochs", str(prior_epochs),
        "--samples_per_type", str(samples_per_type),
        "--save_dir", checkpoints_dir
    ]
    print(f"\n[Step 1/6] Training VQ-VAE & Prior (N={N})...")
    subprocess.run(vq_train_cmd, check=True)
    
    # 2. Train Metric-Aligned Beta-VAE
    aligned_train_cmd = [
        python_bin, "train_metric.py",
        "--N", str(N),
        "--epochs", str(vae_epochs),
        "--beta", str(beta),
        "--gamma", str(gamma),
        "--samples_per_type", str(samples_per_type),
        "--save_dir", checkpoints_dir
    ]
    print(f"\n[Step 2/6] Training Aligned Beta-VAE (N={N}, beta={beta}, gamma={gamma})...")
    subprocess.run(aligned_train_cmd, check=True)
    
    # 3. Train Unaligned Beta-VAE and Evaluate Cascade Router comparison
    cascade_cmd = [
        python_bin, "evaluate_cascade.py",
        "--N", str(N),
        "--beta_vae_epochs", str(vae_epochs),
        "--beta", str(beta),
        "--samples_per_type", str(samples_per_type),
        "--save_dir", checkpoints_dir
    ]
    print(f"\n[Step 3/6] Training Unaligned VAE and Evaluating Cascade Router (N={N})...")
    subprocess.run(cascade_cmd, check=True)
    
    # 4. Generate Latent Space Topologies
    latent_cmd = [
        python_bin, "visualize_latent.py"
    ]
    print(f"\n[Step 4/6] Generating Latent Space Visualizations (N={N})...")
    subprocess.run(latent_cmd, check=True)
    
    # 5. Run Latent Space Interpolation (Climbing the Tree)
    interpolate_cmd = [
        python_bin, "interpolate.py"
    ]
    print(f"\n[Step 5/6] Running Latent Space Interpolation Path (N={N})...")
    subprocess.run(interpolate_cmd, check=True)
    
    # 6. Generate p-adic Tree Branching plots
    visualize_cmd = [
        python_bin, "visualize.py"
    ]
    print(f"\n[Step 6/6] Generating Tree Branching Visualizations (N={N})...")
    subprocess.run(visualize_cmd, check=True)
    
    print("\n======================================================================")
    print("N=64 Pipeline completed successfully!")
    print("======================================================================")

if __name__ == "__main__":
    main()
