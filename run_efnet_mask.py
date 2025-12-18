"""
Multi-Channel ConvSAE Training Script with EfficientNet-B0 Backbone (Masked Loss Variant)

VARIANT: Uses ALL activation channels as input, but computes reconstruction loss
only on the top 85% GradCAM-selected channels (masking others in loss computation).

Key Design:
- Backbone: EfficientNet-B0 pretrained on ImageNet-1k
- Target layer: features[4] (middle layer, ~80 channels, 14×14 resolution)
- Input: ALL activation channels (no pre-masking)
- CSAE: Processes all channels with two-level sparsity
- Loss: Reconstruction error computed ONLY on top 85% GradCAM-selected channels

This approach allows the model to:
1. Learn from richer input (all channels)
2. Focus reconstruction quality on class-discriminative channels
3. Potentially discover useful patterns in "less important" channels

Usage:
    python run_efnet_mask.py
"""

import torch
torch.cuda.init()

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
import sys
import os
import hashlib
sys.path.append('.')
from src.gradcam import GradCAM

# ==========================================
# Multi-Channel ConvSAE Architecture
# ==========================================

class MultiChannelConvSAE(nn.Module):
    """
    Convolutional Sparse Autoencoder for multi-channel input with Two-Level Sparsity.

    Uses 1×1 convolutions to learn sparse features across the channel dimension.
    Each learned feature corresponds to a specific combination of input channels.

    Two-Level Sparsity Mechanism:
        1. Channel-level sparsity (Top-K): For each sample, rank channels by their total
           spatial activation (sum over H×W), then select only top-k channels (hard selection)
        2. Spatial-level sparsity (L1): Among the selected top-k channels, apply L1
           regularization to make each channel's spatial activations sparse

    Args:
        in_channels: Number of input channels (e.g., ~80 for EfficientNet-B0 features[4])
        hidden_dim: Number of sparse features (e.g., 640 for 8× expansion)
        kernel_size: Convolution kernel size (default: 1 for channel-wise features)
        top_k: Number of channels to keep active per sample (default: 10)
    """

    def __init__(self, in_channels: int = 80, hidden_dim: int = 640,
                 kernel_size: int = 1, top_k: int = 10):
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

    def topk_activation(self, x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """
        Apply Top-K channel selection based on spatial activation sum (hard channel sparsity).

        Args:
            x: [B, C, H, W] - Feature activations after ReLU
            threshold: Minimum activation value to keep (default: 0.0, disabled)

        Returns:
            x_topk: [B, C, H, W] - Sparse features with only top-k channels active
        """
        B, C, H, W = x.shape

        # 1. CHANNEL-LEVEL SPARSITY: Rank channels by total spatial activation
        channel_importance = x.sum(dim=[2, 3])  # [B, C]

        # Select top-k channels based on importance (per sample)
        topk_vals, topk_indices = torch.topk(channel_importance, k=self.top_k, dim=1)  # [B, k]

        # Apply threshold to importance scores if needed (optional)
        if threshold > 0:
            threshold_mask = topk_vals > threshold  # [B, k]
        else:
            threshold_mask = None

        # Create channel selection mask [B, C]
        channel_mask = torch.zeros(B, C, device=x.device, dtype=torch.bool)
        channel_mask.scatter_(1, topk_indices, True)

        # Apply threshold mask if specified
        if threshold_mask is not None:
            for b in range(B):
                valid_channels = topk_indices[b][threshold_mask[b]]
                temp_mask = torch.zeros(C, device=x.device, dtype=torch.bool)
                temp_mask[valid_channels] = True
                channel_mask[b] = temp_mask

        # 2. SPATIAL-LEVEL: Keep original spatial activations for selected channels
        channel_mask_4d = channel_mask.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        result = x * channel_mask_4d.float()  # [B, C, H, W]

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
            sparse_features = self.topk_activation(features)
        else:
            sparse_features = features

        # Decode: Reconstruct input
        reconstruction = self.decoder(sparse_features)

        return reconstruction, sparse_features

    def normalize_decoder_weights(self):
        """
        Normalize decoder weights to have unit norm per feature.
        Standard practice in SAE to prevent scale ambiguity.
        """
        with torch.no_grad():
            weight = self.decoder.weight.data
            norm = weight.norm(p=2, dim=(0, 2, 3), keepdim=True).clamp(min=1e-8)
            self.decoder.weight.data = weight / norm


class LateralInhibitionLoss(nn.Module):
    """Penalizes neighboring features from activating together."""

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape
        feat_center = features[:, :, 1:-1, 1:-1]
        feat_left = features[:, :, 1:-1, :-2]
        feat_right = features[:, :, 1:-1, 2:]
        feat_up = features[:, :, :-2, 1:-1]
        feat_down = features[:, :, 2:, 1:-1]

        corr = (
            (feat_center * feat_left).mean() +
            (feat_center * feat_right).mean() +
            (feat_center * feat_up).mean() +
            (feat_center * feat_down).mean()
        ) / 4.0

        return corr


class SpatialCompactnessLoss(nn.Module):
    """Spatial Compactness Regularization using Total Variation (TV) loss."""

    def __init__(self):
        super().__init__()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        diff_h = torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :])
        diff_w = torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1])
        tv_loss = diff_h.mean() + diff_w.mean()
        return tv_loss


class FeatureChannelSparsityLoss(nn.Module):
    """Feature-Channel Sparsity Loss."""

    def __init__(self):
        super().__init__()

    def forward(self, encoder_weight: torch.Tensor) -> torch.Tensor:
        weight = encoder_weight.squeeze()
        feature_channel_usage = weight.abs().sum(dim=1)
        sparsity_loss = feature_channel_usage.mean()
        return sparsity_loss


# ==========================================
# EfficientNet-B0 Activation Extractor (NO MASKING)
# ==========================================

class EfficientNetActivationExtractor:
    """
    Extracts activation channels from EfficientNet-B0 middle layer WITHOUT masking.

    VARIANT: Unlike standard version, this extractor:
    1. Collects ALL activation channels (no GradCAM pre-masking)
    2. Also stores GradCAM channel masks for each image
    3. Masks are later used in loss computation during training

    Target Layer: features[4] (middle layer after MBConv block 4)
    - EfficientNet-B0 features[4]: ~80 channels, 14×14 spatial resolution
    """

    def __init__(self, device='cuda', cumulative_threshold=0.85):
        self.device = device
        self.cumulative_threshold = cumulative_threshold

        # Load pretrained EfficientNet-B0
        self.model = models.efficientnet_b0(pretrained=True).to(device)
        self.model.eval()

        # Target layer: features[4] (middle layer)
        self.target_layer = self.model.features[4]

        # Get the actual number of channels from the target layer
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            x = dummy_input
            for i in range(5):  # features[0] through features[4]
                x = self.model.features[i](x)
            self.num_channels = x.shape[1]
            self.spatial_size = x.shape[2]

        print(f"EfficientNet-B0 Target Layer: features[4]")
        print(f"  Output channels: {self.num_channels}")
        print(f"  Spatial resolution: {self.spatial_size}×{self.spatial_size}")

        # GradCAM for channel importance
        self.gradcam = GradCAM(self.model, self.target_layer)

        # Hook for activations
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output.detach()

    def _select_channels_with_gradcam(self, image: torch.Tensor, class_idx: int = None) -> Tuple[torch.Tensor, int]:
        """
        Use GradCAM to select important channels and create a binary mask.

        Args:
            image: [1, 3, 224, 224] - Input image
            class_idx: Target class (None = use predicted class)

        Returns:
            channel_mask: [num_channels] - Binary mask (1 for selected channels, 0 for others)
            num_selected: Number of selected channels
        """
        # Compute GradCAM channel weights
        weights, _, pred_class = self.gradcam.forward(image, class_idx=class_idx, verbose=False)

        # Sort channels by importance (descending)
        sorted_indices = torch.argsort(weights, descending=True)
        sorted_weights = weights[sorted_indices]

        # Normalize to get percentages
        total_score = sorted_weights.sum()
        if total_score > 0:
            cumsum = torch.cumsum(sorted_weights / total_score, dim=0)
            num_selected = (cumsum < self.cumulative_threshold).sum().item() + 1
            num_selected = min(num_selected, len(sorted_indices))
        else:
            num_selected = max(1, int(0.1 * len(sorted_indices)))

        # Create binary mask
        channel_mask = torch.zeros(self.num_channels, dtype=torch.bool, device=self.device)
        selected_channels = sorted_indices[:num_selected]
        channel_mask[selected_channels] = True

        return channel_mask, num_selected

    def _get_cache_filename(self, data_loader: DataLoader, normalize: bool) -> str:
        """Generate a unique cache filename based on dataset and configuration."""
        config_str = f"efnet_features4_nomask_threshold{self.cumulative_threshold}_normalize{normalize}_nsamples{len(data_loader.dataset)}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        cache_dir = "cache_activations"
        os.makedirs(cache_dir, exist_ok=True)

        cache_filename = os.path.join(cache_dir, f"efnet_activations_nomask_{config_hash}.pt")
        return cache_filename

    def collect_activation_maps(
        self,
        data_loader: DataLoader,
        normalize: bool = True,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect activation maps WITHOUT masking (keep all channels).
        Also collect GradCAM channel selection masks for each image.

        VARIANT: Returns:
        1. X: Full activation maps [N, num_channels, H, W] (NO masking applied)
        2. masks: GradCAM channel masks [N, num_channels] (for loss computation)

        Args:
            data_loader: DataLoader with (image, label) pairs
            normalize: Apply robust normalization (99th percentile)
            use_cache: Whether to use cached activations if available (default: True)

        Returns:
            X: [N, num_channels, H, W] - Full activation maps (all channels kept)
            masks: [N, num_channels] - Binary masks for GradCAM-selected channels
        """
        # Check for cached activations
        if use_cache:
            cache_filename = self._get_cache_filename(data_loader, normalize)
            if os.path.exists(cache_filename):
                print(f"\n{'='*80}")
                print(f"Loading cached activations from: {cache_filename}")
                print(f"{'='*80}")
                cached_data = torch.load(cache_filename)
                X = cached_data['activations']
                masks = cached_data['masks']
                print(f"\nLoaded {X.shape[0]} activation maps from cache:")
                print(f"  Activations shape: {X.shape}")
                print(f"  Masks shape: {masks.shape}")
                print(f"  Range: [{X.min():.4f}, {X.max():.4f}]")
                print(f"  Mean: {X.mean():.4f}, Std: {X.std():.4f}")
                return X, masks
            else:
                print(f"\nCache file not found. Will save to: {cache_filename}")

        all_activations = []
        all_masks = []
        channel_selection_stats = []

        print(f"Collecting activation maps from EfficientNet-B0 features[4] WITHOUT masking...")
        print(f"  GradCAM threshold: {self.cumulative_threshold * 100:.0f}% (for mask generation)")

        for images, labels in tqdm(data_loader, desc="Extracting activations"):
            batch_activations = []
            batch_masks = []

            for i in range(images.size(0)):
                image = images[i:i+1].to(self.device)

                # Forward pass to get activations
                with torch.no_grad():
                    _ = self.model(image)
                    activations = self.activations.clone()  # [1, num_channels, H, W]

                # Get GradCAM-based channel mask (requires gradients)
                channel_mask, num_selected = self._select_channels_with_gradcam(image)
                channel_selection_stats.append(num_selected)

                # VARIANT: Do NOT apply mask to activations (keep all channels)
                batch_activations.append(activations.cpu())
                batch_masks.append(channel_mask.cpu())

            # Concatenate batch
            all_activations.append(torch.cat(batch_activations, dim=0))
            all_masks.append(torch.stack(batch_masks, dim=0))

        # Concatenate all batches
        X = torch.cat(all_activations, dim=0)  # [N, num_channels, H, W]
        masks = torch.cat(all_masks, dim=0)  # [N, num_channels]

        # Print statistics
        avg_selected = np.mean(channel_selection_stats)
        std_selected = np.std(channel_selection_stats)
        print(f"\nChannel selection statistics (for masks):")
        print(f"  Average channels selected: {avg_selected:.1f} ± {std_selected:.1f} (out of {self.num_channels})")
        print(f"  Min: {min(channel_selection_stats)}, Max: {max(channel_selection_stats)}")
        print(f"  These masks will be used for loss computation during training")

        print(f"\nCollected {X.shape[0]} activation maps:")
        print(f"  Shape: {X.shape}")
        print(f"  ALL CHANNELS KEPT (no masking applied)")
        print(f"  Range before normalization: [{X.min():.4f}, {X.max():.4f}]")
        print(f"  Mean: {X.mean():.4f}, Std: {X.std():.4f}")

        # Robust normalization (applied to ALL channels)
        if normalize:
            print("\nApplying robust normalization (per-channel, ALL channels)...")

            for c in range(X.shape[1]):
                channel_data = X[:, c, :, :]

                flat = channel_data.flatten()
                non_zero_flat = flat[flat > 1e-8]
                if len(non_zero_flat) > 0:
                    scale_factor = torch.quantile(non_zero_flat, 0.99)

                    if scale_factor > 1e-8:
                        channel_data = torch.clamp(channel_data, min=0.0, max=scale_factor)
                        X[:, c, :, :] = channel_data / (scale_factor + 1e-8)

            print(f"  Normalized range: [{X.min():.4f}, {X.max():.4f}]")
            print(f"  Mean: {X.mean():.4f}, Std: {X.std():.4f}")

        # Save to cache
        if use_cache:
            cache_filename = self._get_cache_filename(data_loader, normalize)
            print(f"\nSaving activations and masks to cache: {cache_filename}")
            torch.save({'activations': X, 'masks': masks}, cache_filename)
            print(f"✓ Activations and masks cached successfully!")

        return X, masks


# ==========================================
# Masked Reconstruction Loss
# ==========================================

def masked_reconstruction_loss(reconstruction: torch.Tensor,
                               target: torch.Tensor,
                               masks: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE reconstruction loss only on GradCAM-selected channels.

    Args:
        reconstruction: [B, C, H, W] - Reconstructed activation maps
        target: [B, C, H, W] - Target activation maps
        masks: [B, C] - Binary masks (1 for selected channels, 0 for others)

    Returns:
        loss: Scalar - Masked MSE loss
    """
    # Expand masks to [B, C, 1, 1] for broadcasting
    masks_4d = masks.unsqueeze(2).unsqueeze(3).float()

    # Compute squared error
    squared_error = (reconstruction - target) ** 2

    # Apply mask
    masked_squared_error = squared_error * masks_4d

    # Compute mean over selected elements
    num_selected = masks.sum(dim=1, keepdim=True).float()
    num_selected = num_selected.clamp(min=1.0)

    loss_per_sample = masked_squared_error.sum(dim=(1, 2, 3)) / (num_selected.squeeze() * reconstruction.shape[2] * reconstruction.shape[3])
    loss = loss_per_sample.mean()

    return loss


# ==========================================
# Visualization Functions
# ==========================================

def plot_training_logs(logs: Dict[str, List], save_path: str = 'multichannel_csae_efnet_mask_logs.png'):
    """Plot training metrics."""
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Multi-Channel ConvSAE Training (EfficientNet-B0 - Masked Loss Variant)',
                 fontsize=14, fontweight='bold')

    # Row 1: Main losses
    axs[0, 0].plot(logs["recon_loss"], color='blue', linewidth=1.5)
    axs[0, 0].set_title("Masked Reconstruction Loss")
    axs[0, 0].set_ylabel("MSE (masked)")
    axs[0, 0].set_xlabel("Batch")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(logs["l1_loss"], color='green', linewidth=1.5)
    axs[0, 1].set_title("L1 Sparsity Loss")
    axs[0, 1].set_ylabel("L1")
    axs[0, 1].set_xlabel("Batch")
    axs[0, 1].grid(True, alpha=0.3)

    axs[0, 2].plot(logs["channel_sparsity_loss"], color='purple', linewidth=1.5)
    axs[0, 2].set_title("Channel Sparsity Loss")
    axs[0, 2].set_ylabel("L1 per feature")
    axs[0, 2].set_xlabel("Batch")
    axs[0, 2].grid(True, alpha=0.3)

    # Row 2: Regularization losses
    axs[1, 0].plot(logs["lateral_loss"], color='orange', linewidth=1.5)
    axs[1, 0].set_title("Lateral Inhibition Loss")
    axs[1, 0].set_ylabel("Correlation")
    axs[1, 0].set_xlabel("Batch")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(logs["compact_loss"], color='red', linewidth=1.5)
    axs[1, 1].set_title("Spatial Compactness Loss (TV)")
    axs[1, 1].set_ylabel("Total Variation")
    axs[1, 1].set_xlabel("Batch")
    axs[1, 1].grid(True, alpha=0.3)

    axs[1, 2].plot(logs["active_pct"], color='teal', linewidth=1.5)
    axs[1, 2].set_title("Active Channels %")
    axs[1, 2].set_ylabel("Percent (%)")
    axs[1, 2].set_xlabel("Batch")
    axs[1, 2].set_ylim(0, 10)
    axs[1, 2].grid(True, alpha=0.3)

    # Row 3: Summary metrics
    axs[2, 0].plot(logs["total_loss"], color='black', linewidth=2)
    axs[2, 0].set_title("Total Loss")
    axs[2, 0].set_ylabel("Loss")
    axs[2, 0].set_xlabel("Batch")
    axs[2, 0].grid(True, alpha=0.3)

    axs[2, 1].plot(logs["recon_loss"], label='Recon (masked)', alpha=0.7)
    axs[2, 1].plot(logs["l1_loss"], label='L1', alpha=0.7)
    axs[2, 1].plot(logs["lateral_loss"], label='Lateral', alpha=0.7)
    axs[2, 1].plot(logs["compact_loss"], label='Compact', alpha=0.7)
    axs[2, 1].plot(logs["channel_sparsity_loss"], label='Ch-Sp', alpha=0.7)
    axs[2, 1].set_title("Loss Components (Log Scale)")
    axs[2, 1].set_ylabel("Loss")
    axs[2, 1].set_xlabel("Batch")
    axs[2, 1].set_yscale('log')
    axs[2, 1].legend(fontsize=7)
    axs[2, 1].grid(True, alpha=0.3)

    axs[2, 2].scatter(logs["channel_sparsity_loss"], logs["recon_loss"],
                     c=range(len(logs["recon_loss"])), cmap='viridis',
                     alpha=0.5, s=5)
    axs[2, 2].set_title("Reconstruction vs Channel Sparsity")
    axs[2, 2].set_xlabel("Channel Sparsity Loss")
    axs[2, 2].set_ylabel("Masked Reconstruction Loss")
    axs[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training logs saved to {save_path}")
    plt.close()


def visualize_learned_features(model: MultiChannelConvSAE,
                               num_features: int = 32,
                               save_path: str = 'multichannel_csae_efnet_mask_features.png'):
    """Visualize learned decoder features."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Multi-Channel ConvSAE Learned Features (EfficientNet-B0 - Masked Loss)', fontsize=14, fontweight='bold')

    decoder_weights = model.decoder.weight.detach().cpu().squeeze()

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

    # 2. Feature L2 norms
    feature_norms = decoder_weights.norm(dim=0).numpy()
    n_show = min(num_features, len(feature_norms))
    axes[0, 1].bar(range(n_show), feature_norms[:n_show], color='green', alpha=0.7)
    axes[0, 1].set_title(f'Feature Magnitudes (Top {n_show})')
    axes[0, 1].set_xlabel('Feature Index')
    axes[0, 1].set_ylabel('L2 Norm')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Channel usage distribution
    channel_importance = decoder_weights.abs().sum(dim=1).numpy()
    axes[1, 0].bar(range(len(channel_importance)), channel_importance, color='orange', alpha=0.7)
    axes[1, 0].set_title('Input Channel Importance')
    axes[1, 0].set_xlabel(f'Input Channel Index (0-{len(channel_importance)-1})')
    axes[1, 0].set_ylabel('Importance')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Feature sparsity
    feature_sparsity = (decoder_weights.abs() > 1e-3).float().sum(dim=0).numpy()
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
    print("Multi-Channel ConvSAE Training with EfficientNet-B0 Backbone (Masked Loss Variant)")
    print("="*80)
    print("\nVARIANT: Input uses ALL channels, loss computed only on top 85% GradCAM channels")
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
    # 2. EXTRACT ACTIVATION MAPS (NO MASKING)
    # ========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create activation extractor
    extractor = EfficientNetActivationExtractor(device=device)

    # Collect activation maps WITHOUT masking
    print(f"\nExtracting {extractor.num_channels} channels from EfficientNet-B0 features[4]...")
    print("(Caching enabled - subsequent runs will be much faster)")
    X, masks = extractor.collect_activation_maps(
        data_loader,
        normalize=True,
        use_cache=True
    )

    print(f"\nCollected activation maps and masks:")
    print(f"  Activations shape: {X.shape}")
    print(f"  Masks shape: {masks.shape}")
    print(f"  Device: {X.device}")

    # ========================================
    # 3. SETUP CONVSAE TRAINING
    # ========================================
    print("\n" + "="*80)
    print("Setting up Multi-Channel ConvSAE training (Masked Loss Variant)...")
    print("="*80)

    # Hyperparameters
    BATCH_SIZE = 64
    INPUT_CHANNELS = extractor.num_channels
    HIDDEN_DIM = INPUT_CHANNELS * 8
    KERNEL_SIZE = 1
    TOP_K = int(HIDDEN_DIM * 0.015)

    # Loss weights
    LAMBDA_L1 = 3.0
    LAMBDA_LAT = 0.01
    LAMBDA_COMPACT = 0.01
    LAMBDA_CHANNEL_SPARSITY = 0.0

    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    EPOCHS = 15

    print(f"\nTraining Configuration:")
    print(f"  Backbone: EfficientNet-B0 (pretrained ImageNet-1k)")
    print(f"  Target layer: features[4] ({INPUT_CHANNELS} channels, {extractor.spatial_size}×{extractor.spatial_size} resolution)")
    print(f"  Input Channels: {INPUT_CHANNELS} (ALL channels, no pre-masking)")
    print(f"  Hidden Dim: {HIDDEN_DIM} ({HIDDEN_DIM/INPUT_CHANNELS:.1f}× expansion)")
    print(f"  Kernel Size: {KERNEL_SIZE}×{KERNEL_SIZE}")
    print(f"  Top-K: {TOP_K} ({TOP_K/HIDDEN_DIM*100:.1f}% sparsity)")
    print(f"  Lambda L1: {LAMBDA_L1}")
    print(f"  Lambda Lateral: {LAMBDA_LAT}")
    print(f"  Lambda Compactness: {LAMBDA_COMPACT}")
    print(f"  Lambda Channel Sparsity: {LAMBDA_CHANNEL_SPARSITY}")
    print(f"  Learning Rate: {LR}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"\n  VARIANT: Reconstruction loss computed only on GradCAM-selected channels")

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
    channel_sparsity_loss_fn = FeatureChannelSparsityLoss().to(device)

    # Create DataLoader with masks
    dataset = TensorDataset(X, masks)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Logging
    logs = {
        "total_loss": [],
        "recon_loss": [],
        "l1_loss": [],
        "lateral_loss": [],
        "compact_loss": [],
        "channel_sparsity_loss": [],
        "active_pct": []
    }

    # ========================================
    # 4. TRAINING LOOP
    # ========================================
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80)

    for epoch in range(EPOCHS):
        epoch_metrics = {k: 0 for k in logs.keys()}
        n_batches = 0

        for batch_idx, (batch_acts, batch_masks) in enumerate(train_loader):
            batch_acts = batch_acts.to(device)
            batch_masks = batch_masks.to(device)

            optimizer.zero_grad()

            # Forward pass
            reconstruction, sparse_features = csae_model(batch_acts, use_topk=True)

            # VARIANT: Masked reconstruction loss
            loss_recon = masked_reconstruction_loss(reconstruction, batch_acts, batch_masks)

            # Other losses
            loss_l1 = sparse_features.abs().mean()
            loss_lateral = lat_inhib_loss(sparse_features)
            loss_compact = compact_loss_fn(sparse_features)
            loss_channel_sparsity = channel_sparsity_loss_fn(csae_model.encoder.weight)

            # Combined loss
            loss = (loss_recon +
                   LAMBDA_L1 * loss_l1 +
                   LAMBDA_LAT * loss_lateral +
                   LAMBDA_COMPACT * loss_compact +
                   LAMBDA_CHANNEL_SPARSITY * loss_channel_sparsity)

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
                logs["channel_sparsity_loss"].append(loss_channel_sparsity.item())
                logs["active_pct"].append(active_pct)

                for k in epoch_metrics.keys():
                    epoch_metrics[k] += logs[k][-1]
                n_batches += 1

            # Print progress
            if batch_idx % 20 == 0:
                print(f"\rEpoch {epoch+1}/{EPOCHS} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | Recon (masked): {loss_recon.item():.4f} | "
                      f"Active: {active_pct:.1f}%", end="")

        # Epoch summary
        avg_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}

        print(f"\n[Epoch {epoch+1}/{EPOCHS}] Summary:")
        print(f"  Total Loss: {avg_metrics['total_loss']:.4f}")
        print(f"  Reconstruction (masked): {avg_metrics['recon_loss']:.4f}")
        print(f"  L1 Sparsity: {avg_metrics['l1_loss']:.4f}")
        print(f"  Active Channels: {avg_metrics['active_pct']:.2f}%")
        print("-" * 80)

    print("="*80)
    print("Training Complete!")
    print("="*80)

    # ========================================
    # 5. SAVE MODEL
    # ========================================
    print("\nSaving models...")

    torch.save(csae_model.state_dict(), 'multichannel_csae_efnet_mask_model.pth')
    print("✓ Model state dict: multichannel_csae_efnet_mask_model.pth")

    joblib.dump(csae_model.cpu(), 'multichannel_csae_efnet_mask_model.pkl')
    print("✓ Full model: multichannel_csae_efnet_mask_model.pkl")

    training_info = {
        'config': {
            'backbone': 'EfficientNet-B0',
            'target_layer': 'features[4]',
            'variant': 'masked_loss',
            'input_channels': INPUT_CHANNELS,
            'hidden_dim': HIDDEN_DIM,
            'kernel_size': KERNEL_SIZE,
            'top_k': TOP_K,
            'lambda_l1': LAMBDA_L1,
            'lambda_lateral': LAMBDA_LAT,
            'lambda_compact': LAMBDA_COMPACT,
            'lambda_channel_sparsity': LAMBDA_CHANNEL_SPARSITY,
            'lr': LR,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
        },
        'logs': logs,
        'final_metrics': avg_metrics
    }
    joblib.dump(training_info, 'multichannel_csae_efnet_mask_training_info.pkl')
    print("✓ Training info: multichannel_csae_efnet_mask_training_info.pkl")

    # ========================================
    # 6. VISUALIZATIONS
    # ========================================
    print("\nGenerating visualizations...")

    plot_training_logs(logs, save_path='multichannel_csae_efnet_mask_logs.png')
    csae_model = csae_model.to(device)
    visualize_learned_features(csae_model, num_features=64,
                               save_path='multichannel_csae_efnet_mask_features.png')

    print("\n" + "="*80)
    print("✓ All done! Outputs:")
    print("  - multichannel_csae_efnet_mask_model.pth")
    print("  - multichannel_csae_efnet_mask_model.pkl")
    print("  - multichannel_csae_efnet_mask_training_info.pkl")
    print("  - multichannel_csae_efnet_mask_logs.png")
    print("  - multichannel_csae_efnet_mask_features.png")
    print("="*80)
