import sys, os; _r = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(_r, 'src')); os.chdir(_r)
import os
import subprocess
import sys

def main():
    print("======================================================================")
    print("         p-adic VQ-VAE & Prior Generative Model Pipeline              ")
    print("======================================================================")

    # 1. Paths
    python_bin = os.path.join("venv", "bin", "python")
    if not os.path.exists(python_bin):
        # Fallback to system python if venv doesn't exist
        python_bin = sys.executable
        print(f"Virtual environment python not found at venv/bin/python. Using {python_bin}")
    else:
        print(f"Using virtual environment python: {python_bin}")

    checkpoints_dir = "./checkpoints"
    plots_dir = "./plots"

    # 2. Stage 1 & 2: Train VQ-VAE and Prior
    # We use 10 epochs and 600 samples per type for a fast but high-quality training run.
    vqvae_epochs = 12
    prior_epochs = 12
    samples_per_type = 600

    train_cmd = [
        python_bin, "src/train.py",
        "--vqvae_epochs", str(vqvae_epochs),
        "--prior_epochs", str(prior_epochs),
        "--samples_per_type", str(samples_per_type),
        "--save_dir", checkpoints_dir
    ]

    print(f"\nRunning training script with command:\n{' '.join(train_cmd)}")
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        sys.exit(1)

    # 3. Stage 3: Visualize & Evaluate
    eval_cmd = [
        python_bin, "src/visualize.py"
    ]
    print(f"\nRunning visualization and evaluation with command:\n{' '.join(eval_cmd)}")
    try:
        subprocess.run(eval_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during visualization/evaluation: {e}")
        sys.exit(1)
        
    print("\n======================================================================")
    print("Pipeline completed successfully!")
    print(f"Checkpoints saved to: {os.path.abspath(checkpoints_dir)}")
    print(f"Visualizations saved to: {os.path.abspath(plots_dir)}")
    print("======================================================================")

if __name__ == "__main__":
    main()
