from src.activation import ActivationMapCollector
from src.model import FineTunedModel
from src.convsae import ConvSAE, LateralInhibitionLoss
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm


def plot_training_logs(logs, save_path='csae_training_logs.png'):
    """
    Plots a 2x3 Grid of health metrics.
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle('ConvSAE Training Diagnostics', fontsize=16)

    # 1. Reconstruction (Should go down)
    axs[0, 0].plot(logs["recon_loss"], color='blue')
    axs[0, 0].set_title("Reconstruction Loss (MSE)")
    axs[0, 0].set_ylabel("Error")
    axs[0, 0].grid(True, alpha=0.3)

    # 2. L1 Loss (Should stabilize)
    axs[0, 1].plot(logs["l1_loss"], color='green')
    axs[0, 1].set_title("L1 Loss (Avg Activation Magnitude)")
    axs[0, 1].set_ylabel("Magnitude")
    axs[0, 1].grid(True, alpha=0.3)

    # 3. L0 Loss (Should go down to ~0.05-0.15)
    axs[0, 2].plot(logs["l0_loss"], color='orange')
    axs[0, 2].set_title("L0 Approximation (Target Sparsity)")
    axs[0, 2].set_ylabel("Proportion Active")
    axs[0, 2].axhspan(0.005, 0.10, alpha=0.2, color='green', label='Target (0.5-10%)')
    axs[0, 2].legend()
    axs[0, 2].grid(True, alpha=0.3)

    # 5. Lateral Inhibition (Should go down)
    axs[1, 0].plot(logs["lateral_loss"], color='purple')
    axs[1, 0].set_title("Lateral Inhibition (Blob Penalty)")
    axs[1, 0].set_ylabel("Loss")
    axs[1, 0].grid(True, alpha=0.3)

    # 6. Active Neurons (Should be 5-15%)
    axs[1, 1].plot(logs["active_neurons_pct"], color='red', linewidth=2)
    axs[1, 1].set_title("Active Neurons % (Batch-wise)")
    axs[1, 1].set_ylabel("Percent (%)")
    axs[1, 1].set_ylim(0, 100)

    # Add target range shading
    axs[1, 1].axhspan(5, 15, alpha=0.2, color='green', label='Target Range (5-15%)')
    axs[1, 1].axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='Warning Threshold (30%)')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    # 7. Total Loss (Combined)
    axs[1, 2].plot(logs["total_loss"], color='black', linewidth=1.5)
    axs[1, 2].set_title("Total Loss (All Components)")
    axs[1, 2].set_ylabel("Loss")
    axs[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training logs saved to {save_path}")
    plt.show()


def visualize_learned_features(model, num_features=64, save_path='csae_features.png'):
    """
    Visualize learned encoder/decoder features.

    Args:
        model: Trained ConvSAE model
        num_features: Number of features to visualize
        save_path: Path to save visualization
    """
    # Get decoder weights (features that the model learned)
    decoder_weights = model.decoder.weight.detach().cpu()  # [in_channels, hidden_dim, k, k]
    in_channels = decoder_weights.shape[0]
    hidden_dim = decoder_weights.shape[1]
    kernel_size = decoder_weights.shape[2]

    # For 1x1 convolutions with in_channels=1, visualize as a grid of feature magnitudes
    if in_channels == 1 and kernel_size == 1:
        # Extract all decoder weights [1, hidden_dim, 1, 1] -> [hidden_dim]
        all_weights = decoder_weights.squeeze().numpy()  # [hidden_dim]

        # Show top num_features
        n_features_to_show = min(num_features, hidden_dim)

        # Create a 2D grid visualization of feature weights
        grid_size = int(np.ceil(np.sqrt(n_features_to_show)))

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Pad to square grid
        padded = np.zeros(grid_size * grid_size)
        padded[:n_features_to_show] = all_weights[:n_features_to_show]
        grid = padded.reshape(grid_size, grid_size)

        # Plot heatmap
        im = ax.imshow(grid, cmap='RdBu_r', aspect='auto')
        ax.set_title(f'Decoder Feature Weights (Top {n_features_to_show}/{hidden_dim} features)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Feature Index (column)')
        ax.set_ylabel('Feature Index (row)')

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Weight Magnitude')

    else:
        # For multi-channel or larger kernels, visualize per-feature
        n_features_to_show = min(num_features, decoder_weights.shape[1])
        cols = 8
        rows = int(np.ceil(n_features_to_show / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten()

        for i in range(n_features_to_show):
            # Get the i-th feature's decoder weights across all input channels
            feature_weights = decoder_weights[:, i, :, :].squeeze(-1).squeeze(-1)  # [in_channels]

            # Handle case where in_channels=1 (becomes scalar after squeeze)
            if feature_weights.dim() == 0:
                feature_weights = feature_weights.unsqueeze(0)

            # Reshape to a 2D grid for visualization
            grid_size = int(np.ceil(np.sqrt(feature_weights.shape[0])))
            padded = np.zeros(grid_size * grid_size)
            padded[:feature_weights.shape[0]] = feature_weights.numpy()
            grid = padded.reshape(grid_size, grid_size)

            # Plot
            vmax = max(abs(grid.min()), abs(grid.max()))
            im = axes[i].imshow(grid, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[i].set_title(f'Feature {i}', fontsize=8)
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(n_features_to_show, len(axes)):
            axes[i].axis('off')

        plt.suptitle('Learned ConvSAE Features (Decoder Weights)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Feature visualization saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    # ========================================
    # 1. SETUP DATA
    # ========================================
    print("Setting up Data...")
    data_dir = 'data/imagenette'
    BATCH_SIZE_COLLECTION = 1  # Keep 1 for collection to save VRAM

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
    data_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE_COLLECTION)

    # ========================================
    # 2. COLLECT FEATURE MAPS
    # ========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = FineTunedModel(num_classes=10).to(device)
    model.load_state_dict(torch.load('weights/finetune_weights.pth'))
    target_layer = model.feature_extractor[5]

    collector = ActivationMapCollector(model, target_layer, device=device)

    # Collect data
    print("Collecting Activation Maps...")
    X = collector.collect_maps(data_loader, output_tensor=True, device=device)
    print(f"Collected {len(X)} activation maps with shape {X.shape}")

    # ========================================
    # 3. ROBUST NORMALIZATION
    # ========================================
    print("Applying robust normalization...")
    flat = X.flatten()
    num = min(10_000_000, flat.numel())
    idx = torch.randint(0, flat.numel(), (num,), device=flat.device)
    scale_factor = torch.quantile(flat[idx], 0.99)

    print(f"Robust Scale Factor: {scale_factor:.4f}")
    X = torch.clamp(X, min=0.0, max=scale_factor)
    X = X / scale_factor
    # X = X * 10  # REMOVED: Keeping data in [0, 1] to avoid huge MSE loss

    # Verify data range
    print(f"Data range after normalization: [{X.min():.4f}, {X.max():.4f}]")
    print(f"Data mean: {X.mean():.4f}, std: {X.std():.4f}")

    # OPTIONAL: BLURRING
    # blur = transforms.GaussianBlur(kernel_size=3, sigma=0.5)
    # X = blur(X)

    # ========================================
    # 4. SETUP CONVSAE TRAINING
    # ========================================
    print("\nSetting up ConvSAE training...")

    # Hyperparameters
    BATCH_SIZE = 256
    INPUT_CHANNELS = X.shape[1]  # Should be 1 (single-channel activation maps)
    HIDDEN_DIM = 4096
    KERNEL_SIZE = 1  # 1x1 convolution for spatial sparsity
    LAMBDA_L1 = 0.0  # DISABLED - focus purely on reconstruction to debug
    LAMBDA_LAT = 0.0  # DISABLED - testing if lateral inhibition causes dead neurons
    LR = 3e-4
    EPOCHS = 10

    print(f"Training Configuration:")
    print(f"  Input Channels: {INPUT_CHANNELS}")
    print(f"  Hidden Dim: {HIDDEN_DIM}")
    print(f"  Kernel Size: {KERNEL_SIZE}")
    print(f"  Lambda L1: {LAMBDA_L1}")
    print(f"  Lambda Lateral: {LAMBDA_LAT}")
    print(f"  Learning Rate: {LR}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")

    # Create ConvSAE model
    # Note: Input channels should match the collected activation maps
    # The collected maps have shape [N, 1, H, W] so in_channels=1
    csae_model = ConvSAE(
        in_channels=INPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        kernel_size=KERNEL_SIZE
    ).to(device)

    # Initialize encoder bias to very large positive value to force activations
    with torch.no_grad():
        csae_model.encoder_bias.data.fill_(1.0)  # Increased to 1.0 for debugging

    optimizer = optim.Adam(csae_model.parameters(), lr=LR)
    lat_inhib_loss = LateralInhibitionLoss().to(device)

    # Create DataLoader for training
    dataset = TensorDataset(X)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Logging dictionary
    logs = {
        "total_loss": [],
        "recon_loss": [],
        "l1_loss": [],
        "l0_loss": [],
        "lateral_loss": [],
        "active_neurons_pct": []
    }

    # ========================================
    # 5. TRAINING LOOP
    # ========================================
    print("\nStarting ConvSAE Training...")
    print("=" * 70)

    for epoch in range(EPOCHS):
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_l1_loss = 0
        epoch_lat_loss = 0
        epoch_active_pct = 0
        n_batches = 0

        for batch_idx, (batch_acts,) in enumerate(train_loader):
            batch_acts = batch_acts.to(device)

            optimizer.zero_grad()

            # Forward pass
            recon, acts = csae_model(batch_acts)

            # Compute losses
            loss_recon = F.mse_loss(recon, batch_acts)

            # L1 Loss: Average activation magnitude
            loss_l1 = acts.abs().mean()

            # L0 Approximation: Penalize the NUMBER of active neurons
            # Use smooth approximation: sigmoid makes it differentiable
            # This directly targets the percentage of active neurons
            k = 20  # Steepness parameter (increased from 10 for sharper penalty)
            l0_approx = torch.sigmoid(k * acts).mean()  # Proportion of "active" neurons

            # Lateral inhibition
            loss_lat = lat_inhib_loss(acts)

            # Combined loss - removed L0 penalty to prevent dead neurons
            # Focus on reconstruction + L1 sparsity only
            loss = loss_recon + (LAMBDA_L1 * loss_l1) + (LAMBDA_LAT * loss_lat)

            # Backward pass
            loss.backward()
            optimizer.step()
            csae_model.normalize_decoder_weights()

            # Collect metrics
            with torch.no_grad():
                active_mask = (acts > 0).float()
                active_pct = active_mask.mean().item() * 100

                logs["total_loss"].append(loss.item())
                logs["recon_loss"].append(loss_recon.item())
                logs["l1_loss"].append(loss_l1.item())
                logs["l0_loss"].append(l0_approx.item())
                logs["lateral_loss"].append(loss_lat.item())
                logs["active_neurons_pct"].append(active_pct)

                epoch_total_loss += loss.item()
                epoch_recon_loss += loss_recon.item()
                epoch_l1_loss += loss_l1.item()
                epoch_lat_loss += loss_lat.item()
                epoch_active_pct += active_pct
                n_batches += 1

            # Print progress every 20 batches
            if batch_idx % 20 == 0:
                print(f"\rEpoch {epoch+1}/{EPOCHS} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | Recon: {loss_recon.item():.4f} | "
                      f"L1: {loss_l1.item():.4f} | Lat: {loss_lat.item():.4f} | "
                      f"Active: {active_pct:.2f}%", end="")

        # Epoch summary
        avg_total = epoch_total_loss / n_batches
        avg_recon = epoch_recon_loss / n_batches
        avg_l1 = epoch_l1_loss / n_batches
        avg_lat = epoch_lat_loss / n_batches
        avg_active = epoch_active_pct / n_batches

        # Sparsity warning
        sparsity_warning = ""
        if avg_active > 10:
            sparsity_warning = " ⚠️  WARNING: Too many active neurons! Increase LAMBDA_L1"
        elif avg_active < 0.5:
            sparsity_warning = " ⚠️  WARNING: Too few active neurons! Model may be dead. Decrease LAMBDA_L1"
        elif 0.5 <= avg_active <= 10:
            sparsity_warning = " ✓ Good sparsity level"

        # Reconstruction quality check
        recon_warning = ""
        if avg_recon > 10.0:  # With data in [0, 10], MSE > 10 means poor reconstruction
            recon_warning = " ⚠️  High reconstruction loss - check model capacity"
        elif avg_recon < 0.5:  # Very good reconstruction
            recon_warning = " ✓ Excellent reconstruction"
        elif avg_recon < 2.0:  # Good reconstruction
            recon_warning = " ✓ Good reconstruction"

        print(f"\n[Epoch {epoch+1}/{EPOCHS}] "
              f"Avg Loss: {avg_total:.4f} | "
              f"Recon: {avg_recon:.4f}{recon_warning} | "
              f"L1: {avg_l1:.4f} | "
              f"Lat: {avg_lat:.4f} | "
              f"Active: {avg_active:.2f}%{sparsity_warning}")
        print("-" * 70)

    print("=" * 70)
    print("Training Complete!")

    # ========================================
    # 6. SAVE MODEL
    # ========================================
    print("\nSaving trained ConvSAE model...")

    # Save model state dict
    torch.save(csae_model.state_dict(), 'csae_model.pth')
    print("✓ Model state dict saved to: csae_model.pth")

    # Save entire model using joblib (for easier loading)
    joblib.dump(csae_model.cpu(), 'csae_model.pkl')
    print("✓ Full model saved to: csae_model.pkl")

    # Save training configuration and logs
    training_info = {
        'config': {
            'input_channels': INPUT_CHANNELS,
            'hidden_dim': HIDDEN_DIM,
            'kernel_size': KERNEL_SIZE,
            'lambda_l1': LAMBDA_L1,
            'lambda_lat': LAMBDA_LAT,
            'lr': LR,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
        },
        'logs': logs,
        'final_metrics': {
            'avg_recon_loss': avg_recon,
            'avg_l1_loss': avg_l1,
            'avg_lat_loss': avg_lat,
            'avg_active_pct': avg_active,
        }
    }
    joblib.dump(training_info, 'csae_training_info.pkl')
    print("✓ Training info saved to: csae_training_info.pkl")

    # ========================================
    # 7. VISUALIZATIONS
    # ========================================
    print("\nGenerating visualizations...")

    # Plot training logs
    plot_training_logs(logs, save_path='csae_training_logs.png')

    # Visualize learned features
    csae_model = csae_model.to(device)
    visualize_learned_features(csae_model, num_features=64, save_path='csae_features.png')

    print("\n" + "=" * 70)
    print("✓ All done! Outputs:")
    print("  - csae_model.pth: Model state dict")
    print("  - csae_model.pkl: Full model (joblib)")
    print("  - csae_training_info.pkl: Training config and logs")
    print("  - csae_training_logs.png: Training metrics visualization")
    print("  - csae_features.png: Learned features visualization")
    print("=" * 70)
