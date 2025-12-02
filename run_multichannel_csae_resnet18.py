"""
Multi-Channel ConvSAE Training Script with ResNet18 Backbone

Uses pretrained ResNet18 (ImageNet) to extract ALL 256 activation channels from layer3.
Applies a single Convolutional Sparse Autoencoder with 1×1 convolutions to learn
sparse features across the channel dimension.

Key Design:
- Input: All 256 channels from ResNet18 layer3 (14×14 spatial resolution)
- Architecture: 256 channels → 4096+ sparse features (1×1 conv)
- Goal: Each feature activates for specific combinations of input channels
- Inspired by SAE (Sparse Autoencoder) for LLM interpretability

This approach treats the 256 ResNet channels as a "vocabulary" and learns
interpretable sparse features that combine these channels, similar to how
SAE learns interpretable features from transformer activations.

Usage:
    python run_multichannel_csae_resnet18.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from torchvision import datasets, transforms
import joblib
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm


# ==========================================
# Multi-Channel ConvSAE Architecture
# ==========================================

class MultiChannelConvSAE(nn.Module):
    """
    Convolutional Sparse Autoencoder for multi-channel input with Top-K activation.

    Uses 1×1 convolutions to learn sparse features across the channel dimension.
    Each learned feature corresponds to a specific combination of input channels.

    Key Features:
        - Top-K activation: Only the top-k features activate per spatial position (hard sparsity)
        - Spatial compactness: Encourages localized feature activations

    Architecture:
        - Encoder: Conv2d(in_channels → hidden_dim, kernel_size=1×1)
        - Top-K Activation: z = TopK(ReLU(W_enc * x + b), k)
        - Decoder: Conv2d(hidden_dim → in_channels, kernel_size=1×1)

    Args:
        in_channels: Number of input channels (e.g., 256 for ResNet18 layer3)
        hidden_dim: Number of sparse features (e.g., 4096)
        kernel_size: Convolution kernel size (default: 1 for channel-wise features)
        top_k: Number of features to keep active per spatial position (default: 32)
    """

    def __init__(self, in_channels: int = 256, hidden_dim: int = 4096,
                 kernel_size: int = 1, top_k: int = 32):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.top_k = top_k

        # Encoder: Projects input channels to sparse feature space
        self.encoder = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True
        )

        # Decoder: Reconstructs input from sparse features
        self.decoder = nn.Conv2d(
            hidden_dim,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False  # No bias for decoder (standard in SAE)
        )

        # Initialize encoder with small weights
        nn.init.kaiming_normal_(self.encoder.weight, mode='fan_out', nonlinearity='relu')
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)

        # Initialize decoder weights
        nn.init.kaiming_normal_(self.decoder.weight, mode='fan_in')

    def topk_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Top-K activation: keep only top-k values per spatial position.

        Args:
            x: [B, C, H, W] - Feature activations after ReLU

        Returns:
            x_topk: [B, C, H, W] - Sparse features with only top-k active
        """
        B, C, H, W = x.shape

        # Reshape to [B, C, H*W] for easier top-k selection
        x_flat = x.view(B, C, H * W)  # [B, C, H*W]

        # Get top-k values and indices per spatial position
        # We want top-k across the channel dimension (dim=1) for each spatial position
        topk_vals, topk_indices = torch.topk(x_flat, k=self.top_k, dim=1)  # [B, k, H*W]

        # Create sparse tensor with only top-k values
        result = torch.zeros_like(x_flat)
        result.scatter_(1, topk_indices, topk_vals)

        # Reshape back to [B, C, H, W]
        result = result.view(B, C, H, W)

        return result

    def forward(self, x: torch.Tensor, use_topk: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.

        Args:
            x: [B, in_channels, H, W] - Input activation maps
            use_topk: Whether to apply top-k activation (default: True)

        Returns:
            reconstruction: [B, in_channels, H, W] - Reconstructed activation maps
            sparse_features: [B, hidden_dim, H, W] - Sparse feature activations
        """
        # Encode: Project to feature space
        features = self.encoder(x)  # [B, hidden_dim, H, W]

        # Apply ReLU
        features = F.relu(features)

        # Apply Top-K activation for hard sparsity
        if use_topk:
            sparse_features = self.topk_activation(features)  # [B, hidden_dim, H, W]
        else:
            sparse_features = features

        # Decode: Reconstruct input
        reconstruction = self.decoder(sparse_features)  # [B, in_channels, H, W]

        return reconstruction, sparse_features

    def normalize_decoder_weights(self):
        """
        Normalize decoder weights to have unit norm per feature.
        Standard practice in SAE to prevent scale ambiguity.
        """
        with torch.no_grad():
            # Decoder weight shape: [in_channels, hidden_dim, kernel_size, kernel_size]
            weight = self.decoder.weight.data

            # Compute L2 norm per output feature (across input dimension)
            # For 1×1 conv: [in_channels, hidden_dim, 1, 1]
            norm = weight.norm(p=2, dim=(0, 2, 3), keepdim=True).clamp(min=1e-8)

            # Normalize
            self.decoder.weight.data = weight / norm


class LateralInhibitionLoss(nn.Module):
    """
    Penalizes neighboring features from activating together.
    Encourages spatial diversity in feature activations.
    """

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] - Feature activations

        Returns:
            loss: Scalar - Lateral inhibition penalty
        """
        # Compute spatial autocorrelation
        # Apply Gaussian blur to features
        B, C, H, W = features.shape

        # Simple approximation: penalize correlation between neighboring spatial locations
        # Shift features and compute correlation
        feat_center = features[:, :, 1:-1, 1:-1]
        feat_left = features[:, :, 1:-1, :-2]
        feat_right = features[:, :, 1:-1, 2:]
        feat_up = features[:, :, :-2, 1:-1]
        feat_down = features[:, :, 2:, 1:-1]

        # Compute correlation
        corr = (
            (feat_center * feat_left).mean() +
            (feat_center * feat_right).mean() +
            (feat_center * feat_up).mean() +
            (feat_center * feat_down).mean()
        ) / 4.0

        return corr


class SpatialCompactnessLoss(nn.Module):
    """
    Spatial Compactness Regularization using Total Variation (TV) loss.

    Encourages feature activations to form compact, localized spatial regions
    by penalizing spatial gradients. This prevents scattered/noisy activations
    across the feature map.

    Total Variation = sum of absolute differences between neighboring pixels.

    Lower TV = smoother, more compact activations
    Higher TV = scattered, noisy activations
    """

    def __init__(self):
        super().__init__()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Total Variation loss for spatial compactness.

        Args:
            features: [B, C, H, W] - Feature activations

        Returns:
            tv_loss: Scalar - Total variation penalty
        """
        # Compute absolute differences between neighboring spatial positions

        # Horizontal differences: |f(x, y) - f(x+1, y)|
        diff_h = torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :])

        # Vertical differences: |f(x, y) - f(x, y+1)|
        diff_w = torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1])

        # Total variation = sum of all differences
        tv_loss = diff_h.mean() + diff_w.mean()

        return tv_loss


# ==========================================
# ResNet18 Activation Extractor
# ==========================================

class ResNet18ActivationExtractor:
    """
    Extracts ALL 256 activation channels from ResNet18 layer3.

    Unlike the GradCAM-based approach, this extracts the full activation
    tensor without channel selection, preserving all information.
    """

    def __init__(self, device='cuda'):
        self.device = device

        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True).to(device)
        self.model.eval()

        # Target layer: layer3 (outputs 256 channels, 14×14 spatial resolution)
        self.target_layer = self.model.layer3

        # Hook for activations
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output.detach()

    def collect_activation_maps(
        self,
        data_loader: DataLoader,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Collect activation maps from ALL 256 channels.

        Args:
            data_loader: DataLoader with (image, label) pairs
            normalize: Apply robust normalization (99th percentile)

        Returns:
            X: [N, 256, 14, 14] - All activation maps (one per image)
        """
        all_activations = []

        print(f"Collecting activation maps from ResNet18 layer3 (256 channels, 14×14)...")

        for images, labels in tqdm(data_loader, desc="Extracting activations"):
            images = images.to(self.device)

            # Forward pass
            with torch.no_grad():
                _ = self.model(images)

            # Get activations [B, 256, 14, 14]
            activations = self.activations

            # Store
            all_activations.append(activations.cpu())

        # Concatenate all batches
        X = torch.cat(all_activations, dim=0)  # [N, 256, 14, 14]

        print(f"\nCollected {X.shape[0]} activation maps:")
        print(f"  Shape: {X.shape}")
        print(f"  Range before normalization: [{X.min():.4f}, {X.max():.4f}]")
        print(f"  Mean: {X.mean():.4f}, Std: {X.std():.4f}")

        # Robust normalization
        if normalize:
            print("\nApplying robust normalization (per-channel)...")

            # Normalize each channel independently to account for different scales
            for c in range(X.shape[1]):
                channel_data = X[:, c, :, :]

                # 99th percentile clipping
                flat = channel_data.flatten()
                num = min(1_000_000, flat.numel())
                idx = torch.randint(0, flat.numel(), (num,))
                scale_factor = torch.quantile(flat[idx], 0.99)

                if scale_factor > 1e-8:
                    channel_data = torch.clamp(channel_data, min=0.0, max=scale_factor)
                    X[:, c, :, :] = channel_data / (scale_factor + 1e-8)

            print(f"  Normalized range: [{X.min():.4f}, {X.max():.4f}]")
            print(f"  Mean: {X.mean():.4f}, Std: {X.std():.4f}")

        return X


# ==========================================
# Visualization Functions
# ==========================================

def plot_training_logs(logs: Dict[str, List], save_path: str = 'multichannel_csae_logs.png'):
    """Plot training metrics."""
    fig, axs = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle('Multi-Channel ConvSAE Training (ResNet18 - 256 Channels + Top-K)',
                 fontsize=14, fontweight='bold')

    # Reconstruction loss
    axs[0, 0].plot(logs["recon_loss"], color='blue', linewidth=1.5)
    axs[0, 0].set_title("Reconstruction Loss")
    axs[0, 0].set_ylabel("MSE")
    axs[0, 0].set_xlabel("Batch")
    axs[0, 0].grid(True, alpha=0.3)

    # L1 sparsity loss
    axs[0, 1].plot(logs["l1_loss"], color='green', linewidth=1.5)
    axs[0, 1].set_title("L1 Sparsity Loss")
    axs[0, 1].set_ylabel("L1")
    axs[0, 1].set_xlabel("Batch")
    axs[0, 1].grid(True, alpha=0.3)

    # Lateral inhibition loss
    axs[0, 2].plot(logs["lateral_loss"], color='orange', linewidth=1.5)
    axs[0, 2].set_title("Lateral Inhibition Loss")
    axs[0, 2].set_ylabel("Correlation")
    axs[0, 2].set_xlabel("Batch")
    axs[0, 2].grid(True, alpha=0.3)

    # Spatial compactness loss
    axs[0, 3].plot(logs["compact_loss"], color='red', linewidth=1.5)
    axs[0, 3].set_title("Spatial Compactness Loss (TV)")
    axs[0, 3].set_ylabel("Total Variation")
    axs[0, 3].set_xlabel("Batch")
    axs[0, 3].grid(True, alpha=0.3)

    # Active neurons percentage
    axs[1, 0].plot(logs["active_pct"], color='purple', linewidth=1.5)
    axs[1, 0].set_title("Active Neurons % (Top-K enforced)")
    axs[1, 0].set_ylabel("Percent (%)")
    axs[1, 0].set_xlabel("Batch")
    axs[1, 0].set_ylim(0, 10)
    axs[1, 0].grid(True, alpha=0.3)

    # Total loss
    axs[1, 1].plot(logs["total_loss"], color='black', linewidth=2)
    axs[1, 1].set_title("Total Loss")
    axs[1, 1].set_ylabel("Loss")
    axs[1, 1].set_xlabel("Batch")
    axs[1, 1].grid(True, alpha=0.3)

    # Loss components (log scale)
    axs[1, 2].plot(logs["recon_loss"], label='Reconstruction', alpha=0.7)
    axs[1, 2].plot(logs["l1_loss"], label='L1 Sparsity', alpha=0.7)
    axs[1, 2].plot(logs["lateral_loss"], label='Lateral Inhibition', alpha=0.7)
    axs[1, 2].plot(logs["compact_loss"], label='Compactness', alpha=0.7)
    axs[1, 2].set_title("Loss Components (Log Scale)")
    axs[1, 2].set_ylabel("Loss")
    axs[1, 2].set_xlabel("Batch")
    axs[1, 2].set_yscale('log')
    axs[1, 2].legend(fontsize=8)
    axs[1, 2].grid(True, alpha=0.3)

    # Reconstruction vs Compactness trade-off
    axs[1, 3].scatter(logs["compact_loss"], logs["recon_loss"],
                     c=range(len(logs["recon_loss"])), cmap='viridis',
                     alpha=0.5, s=5)
    axs[1, 3].set_title("Reconstruction vs Compactness")
    axs[1, 3].set_xlabel("Compactness Loss")
    axs[1, 3].set_ylabel("Reconstruction Loss")
    axs[1, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training logs saved to {save_path}")
    plt.close()


def visualize_learned_features(model: MultiChannelConvSAE,
                               num_features: int = 32,
                               save_path: str = 'multichannel_csae_features.png'):
    """Visualize learned decoder features."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Multi-Channel ConvSAE Learned Features', fontsize=14, fontweight='bold')

    # Get decoder weights [in_channels=256, hidden_dim, kernel_size, kernel_size]
    decoder_weights = model.decoder.weight.detach().cpu()

    # For 1×1 conv, shape is [256, hidden_dim, 1, 1]
    # Each feature is a 256-dimensional vector
    decoder_weights = decoder_weights.squeeze()  # [256, hidden_dim]

    # 1. Decoder weight distribution
    weights_flat = decoder_weights.flatten().numpy()
    axes[0, 0].hist(weights_flat, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(weights_flat), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(weights_flat):.3f}')
    axes[0, 0].set_title('Decoder Weight Distribution')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Feature L2 norms (how "strong" each feature is)
    feature_norms = decoder_weights.norm(dim=0).numpy()  # [hidden_dim]
    n_show = min(num_features, len(feature_norms))
    axes[0, 1].bar(range(n_show), feature_norms[:n_show], color='green', alpha=0.7)
    axes[0, 1].set_title(f'Feature Magnitudes (Top {n_show})')
    axes[0, 1].set_xlabel('Feature Index')
    axes[0, 1].set_ylabel('L2 Norm')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Channel usage distribution (which input channels are most important)
    channel_importance = decoder_weights.abs().sum(dim=1).numpy()  # [256]
    axes[1, 0].bar(range(256), channel_importance, color='orange', alpha=0.7)
    axes[1, 0].set_title('Input Channel Importance (Sum of Absolute Weights)')
    axes[1, 0].set_xlabel('Input Channel Index (0-255)')
    axes[1, 0].set_ylabel('Importance')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Feature sparsity (how many input channels each feature uses)
    # Count non-zero weights per feature
    feature_sparsity = (decoder_weights.abs() > 1e-3).float().sum(dim=0).numpy()  # [hidden_dim]
    axes[1, 1].hist(feature_sparsity, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(feature_sparsity), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(feature_sparsity):.1f}')
    axes[1, 1].set_title('Feature Sparsity (# Input Channels Used)')
    axes[1, 1].set_xlabel('Number of Active Input Channels')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Feature visualization saved to {save_path}")
    plt.close()


# ==========================================
# Main Training Script
# ==========================================

if __name__ == "__main__":
    # ========================================
    # 1. SETUP DATA
    # ========================================
    print("="*80)
    print("Multi-Channel ConvSAE Training with ResNet18 Backbone")
    print("="*80)
    print("\nSetting up Data...")

    data_dir = 'data/imagenette'
    BATCH_SIZE_COLLECTION = 32

    # Standard ImageNet preprocessing
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
    data_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE_COLLECTION, shuffle=False)

    print(f"Dataset: {len(full_dataset)} images")
    print(f"Classes: {full_dataset.classes}")

    # ========================================
    # 2. EXTRACT ACTIVATION MAPS
    # ========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create activation extractor
    extractor = ResNet18ActivationExtractor(device=device)

    # Collect activation maps (ALL 256 channels)
    print("\nExtracting ALL 256 channels from ResNet18 layer3...")
    X = extractor.collect_activation_maps(
        data_loader,
        normalize=True
    )

    print(f"\nCollected activation maps:")
    print(f"  Shape: {X.shape}")  # [N, 256, 14, 14]
    print(f"  Device: {X.device}")

    # ========================================
    # 3. SETUP CONVSAE TRAINING
    # ========================================
    print("\n" + "="*80)
    print("Setting up Multi-Channel ConvSAE training...")
    print("="*80)

    # Hyperparameters
    BATCH_SIZE = 64
    INPUT_CHANNELS = 256    # All ResNet18 layer3 channels
    HIDDEN_DIM = 4096       # Sparse feature dimension (16× expansion)
    KERNEL_SIZE = 1         # 1×1 conv for channel-wise features
    TOP_K = 32              # Number of active features per spatial position

    # Loss weights
    LAMBDA_L1 = 0.01        # L1 sparsity penalty (reduced since top-k enforces hard sparsity)
    LAMBDA_LAT = 0.01       # Lateral inhibition penalty
    LAMBDA_COMPACT = 0.1    # Spatial compactness penalty (Total Variation)

    LR = 3e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 20

    print(f"\nTraining Configuration:")
    print(f"  Backbone: ResNet18 (pretrained ImageNet)")
    print(f"  Target layer: layer3 (256 channels, 14×14 resolution)")
    print(f"  Input Channels: {INPUT_CHANNELS}")
    print(f"  Hidden Dim: {HIDDEN_DIM} ({HIDDEN_DIM/INPUT_CHANNELS:.1f}× expansion)")
    print(f"  Kernel Size: {KERNEL_SIZE}×{KERNEL_SIZE}")
    print(f"  Top-K: {TOP_K} ({TOP_K/HIDDEN_DIM*100:.1f}% sparsity)")
    print(f"  Lambda L1: {LAMBDA_L1}")
    print(f"  Lambda Lateral: {LAMBDA_LAT}")
    print(f"  Lambda Compactness: {LAMBDA_COMPACT}")
    print(f"  Learning Rate: {LR}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")

    # Create model
    csae_model = MultiChannelConvSAE(
        in_channels=INPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        kernel_size=KERNEL_SIZE,
        top_k=TOP_K
    ).to(device)

    optimizer = optim.Adam(csae_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lat_inhib_loss = LateralInhibitionLoss().to(device)
    compact_loss_fn = SpatialCompactnessLoss().to(device)

    # Create DataLoader
    dataset = TensorDataset(X)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Logging
    logs = {
        "total_loss": [],
        "recon_loss": [],
        "l1_loss": [],
        "lateral_loss": [],
        "compact_loss": [],
        "active_pct": []
    }

    # ========================================
    # 4. TRAINING LOOP
    # ========================================
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80)
    print("\nExpected Behavior:")
    print(f"  • Reconstruction Loss: Decrease to <0.01")
    print(f"  • Active Neurons: ~{TOP_K/HIDDEN_DIM*100:.1f}% (enforced by Top-K={TOP_K})")
    print(f"  • Spatial Compactness: Decrease (smoother feature activations)")
    print(f"  • Feature Sparsity: Each feature uses ~20-50 input channels")
    print(f"  • Top-K ensures exactly {TOP_K} features active per spatial position")
    print("="*80 + "\n")

    for epoch in range(EPOCHS):
        epoch_metrics = {k: 0 for k in logs.keys()}
        n_batches = 0

        for batch_idx, (batch_acts,) in enumerate(train_loader):
            batch_acts = batch_acts.to(device)

            optimizer.zero_grad()

            # Forward pass (with top-k activation)
            reconstruction, sparse_features = csae_model(batch_acts, use_topk=True)

            # Losses
            loss_recon = F.mse_loss(reconstruction, batch_acts)
            loss_l1 = sparse_features.abs().mean()
            loss_lateral = lat_inhib_loss(sparse_features)
            loss_compact = compact_loss_fn(sparse_features)

            # Combined loss
            loss = (loss_recon +
                   LAMBDA_L1 * loss_l1 +
                   LAMBDA_LAT * loss_lateral +
                   LAMBDA_COMPACT * loss_compact)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(csae_model.parameters(), max_norm=1.0)
            optimizer.step()

            # Normalize decoder weights
            csae_model.normalize_decoder_weights()

            # Collect metrics
            with torch.no_grad():
                active_pct = (sparse_features > 0).float().mean().item() * 100

                logs["total_loss"].append(loss.item())
                logs["recon_loss"].append(loss_recon.item())
                logs["l1_loss"].append(loss_l1.item())
                logs["lateral_loss"].append(loss_lateral.item())
                logs["compact_loss"].append(loss_compact.item())
                logs["active_pct"].append(active_pct)

                for k in epoch_metrics.keys():
                    epoch_metrics[k] += logs[k][-1]
                n_batches += 1

            # Print progress
            if batch_idx % 20 == 0:
                print(f"\rEpoch {epoch+1}/{EPOCHS} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | Recon: {loss_recon.item():.4f} | "
                      f"L1: {loss_l1.item():.4f} | Compact: {loss_compact.item():.4f} | "
                      f"Active: {active_pct:.1f}%", end="")

        # Epoch summary
        avg_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}

        print(f"\n[Epoch {epoch+1}/{EPOCHS}] Summary:")
        print(f"  Total Loss: {avg_metrics['total_loss']:.4f}")
        print(f"  Reconstruction: {avg_metrics['recon_loss']:.4f}")
        print(f"  L1 Sparsity: {avg_metrics['l1_loss']:.4f}")
        print(f"  Lateral Inhibition: {avg_metrics['lateral_loss']:.4f}")
        print(f"  Spatial Compactness: {avg_metrics['compact_loss']:.4f}")
        print(f"  Active Neurons: {avg_metrics['active_pct']:.2f}% (Target: {TOP_K/HIDDEN_DIM*100:.1f}%)")

        # Check sparsity target (with top-k, should be close to TOP_K/HIDDEN_DIM)
        expected_pct = TOP_K / HIDDEN_DIM * 100
        if abs(avg_metrics['active_pct'] - expected_pct) > 1.0:
            print(f"  ℹ Info: Active neurons {avg_metrics['active_pct']:.2f}% vs expected {expected_pct:.2f}%")

        print("-" * 80)

    print("="*80)
    print("Training Complete!")
    print("="*80)

    # ========================================
    # 5. SAVE MODEL
    # ========================================
    print("\nSaving models...")

    torch.save(csae_model.state_dict(), 'multichannel_csae_resnet18_model.pth')
    print("✓ Model state dict: multichannel_csae_resnet18_model.pth")

    joblib.dump(csae_model.cpu(), 'multichannel_csae_resnet18_model.pkl')
    print("✓ Full model: multichannel_csae_resnet18_model.pkl")

    training_info = {
        'config': {
            'backbone': 'ResNet18',
            'target_layer': 'layer3',
            'input_channels': INPUT_CHANNELS,
            'hidden_dim': HIDDEN_DIM,
            'kernel_size': KERNEL_SIZE,
            'top_k': TOP_K,
            'lambda_l1': LAMBDA_L1,
            'lambda_lateral': LAMBDA_LAT,
            'lambda_compact': LAMBDA_COMPACT,
            'lr': LR,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
        },
        'logs': logs,
        'final_metrics': avg_metrics
    }
    joblib.dump(training_info, 'multichannel_csae_resnet18_training_info.pkl')
    print("✓ Training info: multichannel_csae_resnet18_training_info.pkl")

    # ========================================
    # 6. VISUALIZATIONS
    # ========================================
    print("\nGenerating visualizations...")

    plot_training_logs(logs, save_path='multichannel_csae_resnet18_logs.png')
    csae_model = csae_model.to(device)
    visualize_learned_features(csae_model, num_features=64,
                               save_path='multichannel_csae_resnet18_features.png')

    print("\n" + "="*80)
    print("✓ All done! Outputs:")
    print("  - multichannel_csae_resnet18_model.pth")
    print("  - multichannel_csae_resnet18_model.pkl")
    print("  - multichannel_csae_resnet18_training_info.pkl")
    print("  - multichannel_csae_resnet18_logs.png")
    print("  - multichannel_csae_resnet18_features.png")
    print("="*80)
