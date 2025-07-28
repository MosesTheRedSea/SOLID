import os
import pandas as pd
import torch
import glob
import matplotlib.pyplot as plt


def graph_baseline():
    save_dir = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/figures/baseline"
    log_files = {
        "PointCloud (DGCNN)": "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/outputs/results/baseline/cloud_results.txt",
        "Depth": "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/outputs/results/baseline/depth_results.txt",
        "RGB": "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/outputs/results/baseline/rgb_results.txt"
    }

    # Define the metrics to plot and their corresponding column names
    metrics = {
        "Accuracy": ("train_acc", "val_acc"),
        "Loss": ("train_loss", "val_loss"),
        "Precision": ("val_precision",),
        "Recall": ("val_recall",)
    }

    # Plotting
    for metric_name, cols in metrics.items():
        plt.figure(figsize=(8, 5))
        for label, path in log_files.items():
            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip().str.lower()  # Clean column names

                if len(cols) == 2:
                    plt.plot(df["epoch"], df[cols[0]], linestyle='--', label=f"{label} - Train")
                    plt.plot(df["epoch"], df[cols[1]], label=f"{label} - Val")
                else:
                    plt.plot(df["epoch"], df[cols[0]], label=f"{label}")
            except Exception as e:
                print(f"Skipping {label} for {metric_name}: {e}")
                continue

        plt.title(f"{metric_name} per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric_name.lower()}_plot.png"))
        plt.close()

def graph_multimodal():
    CHECKPOINT_DIR = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/outputs/checkpoints/multimodal"
    FIGURE_DIR = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/figures/multimodal"
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # Collect all checkpoint files
    ckpt_paths = sorted(
        glob.glob(os.path.join(CHECKPOINT_DIR, "fusion_model_epoch_*.pt")),
        key=lambda p: int(p.split("_")[-1].split(".")[0])
    )

    # Initialize logs
    epochs = []
    accs = []
    precisions = []
    recalls = []
    f1s = []

    # Parse each checkpoint
    for path in ckpt_paths:
        data = torch.load(path, map_location='cpu', weights_only=False)
        epoch = data['epoch']
        metrics = data.get('metrics', {})

        epochs.append(epoch)
        accs.append(metrics.get('test_accuracy', 0.0))
        precisions.append(metrics.get('test_precision', 0.0))
        recalls.append(metrics.get('test_recall', 0.0))
        f1s.append(metrics.get('test_f1_macro', 0.0))

    # --- Plotting ---
    def plot_metric(metric_values, title, ylabel, filename):
        plt.figure()
        plt.plot(epochs, metric_values, marker='o')
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(FIGURE_DIR, filename)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
        plt.close()

    plot_metric(accs, "Test Accuracy over Epochs", "Accuracy", "fusion_accuracy.png")
    plot_metric(precisions, "Test Precision over Epochs", "Precision", "fusion_precision.png")
    plot_metric(recalls, "Test Recall over Epochs", "Recall", "fusion_recall.png")
    plot_metric(f1s, "Test F1 Score over Epochs", "F1 Score", "fusion_f1.png")

if __name__ == "__main__":

    #graph_baseline()

    graph_multimodal()