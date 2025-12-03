"""
Multi-Channel ConvSAE Training Script with ResNet50 Backbone (Two-Level Sparsity)

Uses pretrained ResNet50 (ImageNet) to extract ALL 1024 activation channels from layer3.
Applies a single Convolutional Sparse Autoencoder with 1×1 convolutions to learn
sparse features across the channel dimension.

Key Design:
- Input: All 1024 channels from ResNet50 layer3 (14×14 spatial resolution)
- Architecture: 1024 channels → 8192+ sparse features (1×1 conv)
- Two-Level Sparsity:
  1. Channel-level: Top-K selection based on sum of spatial activations (H×W)
  2. Spatial-level: L1 regularization within selected channels for spatial sparsity
- Goal: Each feature activates for specific combinations of input channels
- Inspired by SAE (Sparse Autoencoder) for LLM interpretability

This approach treats the 1024 ResNet channels as a "vocabulary" and learns
interpretable sparse features that combine these channels. The two-level sparsity
ensures that (1) only the most important channels activate per sample, and
(2) within those channels, activations are spatially sparse.

Usage:
    python run_multichannel_csae_resnet50.py
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
sys.path.append('.')
from src.gradcam import GradCAM
from visualized_resnet50 import visualize_multichannel_sae_r50

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

    Key Features:
        - Top-K channel selection: Only the top-k channels activate per sample (hard channel sparsity)
        - L1 spatial regularization: Encourages sparse spatial activations within selected channels
        - Spatial compactness: Optional TV loss for localized feature activations

    Architecture:
        - Encoder: Conv2d(in_channels → hidden_dim, kernel_size=1×1)
        - Channel Selection: z_channels = TopK_channels(ReLU(W_enc * x + b), k)
        - Spatial Sparsity: Applied via L1 regularization during training
        - Decoder: Conv2d(hidden_dim → in_channels, kernel_size=1×1)

    Args:
        in_channels: Number of input channels (e.g., 1024 for ResNet50 layer3)
        hidden_dim: Number of sparse features (e.g., 8192)
        kernel_size: Convolution kernel size (default: 1 for channel-wise features)
        top_k: Number of channels to keep active per sample (default: 64)
    """

    def __init__(self, in_channels: int = 1024, hidden_dim: int = 8192,
                 kernel_size: int = 1, top_k: int = 64):
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

        Two-level sparsity mechanism:
        1. Channel-level: Select top-k channels based on sum of spatial activations (H×W)
        2. Spatial-level: Keep original spatial pattern for selected channels
                         (L1 regularization in training loop enforces spatial sparsity)

        Args:
            x: [B, C, H, W] - Feature activations after ReLU
            threshold: Minimum activation value to keep (default: 0.0, disabled)

        Returns:
            x_topk: [B, C, H, W] - Sparse features with only top-k channels active
        """
        B, C, H, W = x.shape

        # 1. CHANNEL-LEVEL SPARSITY: Rank channels by total spatial activation
        # Sum over spatial dimensions (H×W) for each channel
        channel_importance = x.sum(dim=[2, 3])  # [B, C] - sum over H×W

        # Select top-k channels based on importance (per sample)
        topk_vals, topk_indices = torch.topk(channel_importance, k=self.top_k, dim=1)  # [B, k]

        # Apply threshold to importance scores if needed (optional)
        if threshold > 0:
            # Create mask for channels above threshold
            threshold_mask = topk_vals > threshold  # [B, k]
        else:
            threshold_mask = None

        # Create channel selection mask [B, C]
        channel_mask = torch.zeros(B, C, device=x.device, dtype=torch.bool)
        channel_mask.scatter_(1, topk_indices, True)  # Mark top-k channels as True

        # Apply threshold mask if specified
        if threshold_mask is not None:
            # Zero out channels that didn't meet threshold
            for b in range(B):
                valid_channels = topk_indices[b][threshold_mask[b]]
                temp_mask = torch.zeros(C, device=x.device, dtype=torch.bool)
                temp_mask[valid_channels] = True
                channel_mask[b] = temp_mask

        # 2. SPATIAL-LEVEL: Keep original spatial activations for selected channels
        # Expand mask to [B, C, H, W] for broadcasting
        channel_mask_4d = channel_mask.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]

        # Zero out non-selected channels (keep full spatial pattern for selected channels)
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


class FeatureChannelSparsityLoss(nn.Module):
    """
    Feature-Channel Sparsity Loss.

    Encourages each learned feature to respond to only a few of the 1024 input channels.
    This makes features more interpretable by ensuring they correspond to specific
    combinations of input channels, rather than using all channels.

    Similar to how SAE features in LLM interpretability are encouraged to activate
    for specific token patterns, we encourage features to activate for specific
    channel combinations.

    Implementation:
        For each feature (row in encoder weight matrix), compute L1 norm across
        input channels. Penalize features that have large L1 norms (use many channels).
    """

    def __init__(self):
        super().__init__()

    def forward(self, encoder_weight: torch.Tensor) -> torch.Tensor:
        """
        Compute feature-channel sparsity loss.

        Args:
            encoder_weight: [out_channels, in_channels, k, k] - Encoder weights
                           For 1×1 conv: [8192, 1024, 1, 1]

        Returns:
            sparsity_loss: Scalar - Feature-channel sparsity penalty
        """
        # Squeeze spatial dimensions for 1×1 conv: [8192, 1024]
        weight = encoder_weight.squeeze()

        # For each feature (row), compute L1 norm across input channels
        # This measures how many input channels each feature uses
        feature_channel_usage = weight.abs().sum(dim=1)  # [8192]

        # Penalize features that use many channels
        # Mean over all features
        sparsity_loss = feature_channel_usage.mean()

        return sparsity_loss


# ==========================================
# ResNet50 Activation Extractor
# ==========================================

class ResNet50ActivationExtractor:
    """
    Extracts activation channels from ResNet50 layer3 with GradCAM-based selection.

    For each image:
    1. Compute GradCAM scores for all 1024 channels at layer3
    2. Select channels with cumulative score ≥ 80%
    3. Zero-out unselected channels but keep them (maintain channel positions)

    This approach preserves spatial structure while focusing on class-discriminative channels.
    """

    def __init__(self, device='cuda', cumulative_threshold=0.8):
        self.device = device
        self.cumulative_threshold = cumulative_threshold

        # Load pretrained ResNet50
        self.model = models.resnet50(pretrained=True).to(device)
        self.model.eval()

        # Target layer: layer3 (outputs 1024 channels, 14×14 spatial resolution)
        self.target_layer = self.model.layer3

        # GradCAM for channel importance
        self.gradcam = GradCAM(self.model, self.target_layer)

        # Hook for activations
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output.detach()

    def _select_channels_with_gradcam(self, image: torch.Tensor, class_idx: int = None) -> torch.Tensor:
        """
        Use GradCAM to select important channels and create a binary mask.

        Args:
            image: [1, 3, 224, 224] - Input image
            class_idx: Target class (None = use predicted class)

        Returns:
            channel_mask: [1024] - Binary mask (1 for selected channels, 0 for others)
        """
        # Compute GradCAM channel weights
        weights, _, pred_class = self.gradcam.forward(image, class_idx=class_idx, verbose=False)

        # weights: [1024] - GradCAM importance scores (already ReLU'd)

        # Sort channels by importance (descending)
        sorted_indices = torch.argsort(weights, descending=True)
        sorted_weights = weights[sorted_indices]

        # Normalize to get percentages
        total_score = sorted_weights.sum()
        if total_score > 0:
            cumsum = torch.cumsum(sorted_weights / total_score, dim=0)

            # Find number of channels needed for cumulative_threshold
            num_selected = (cumsum < self.cumulative_threshold).sum().item() + 1
            num_selected = min(num_selected, len(sorted_indices))
        else:
            # If all weights are zero, select top 10% channels
            num_selected = max(1, int(0.1 * len(sorted_indices)))

        # Create binary mask
        channel_mask = torch.zeros(1024, dtype=torch.bool, device=self.device)
        selected_channels = sorted_indices[:num_selected]
        channel_mask[selected_channels] = True

        return channel_mask, num_selected

    def collect_activation_maps(
        self,
        data_loader: DataLoader,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Collect activation maps with GradCAM-based channel selection.

        For each image:
        - Compute GradCAM scores for all 1024 channels
        - Select channels with cumulative score ≥ threshold (default 80%)
        - Zero-out unselected channels but keep tensor shape [1024, 14, 14]

        Args:
            data_loader: DataLoader with (image, label) pairs
            normalize: Apply robust normalization (99th percentile)

        Returns:
            X: [N, 1024, 14, 14] - Activation maps with selective zero-masking
        """
        all_activations = []
        channel_selection_stats = []

        print(f"Collecting activation maps from ResNet50 layer3 with GradCAM selection...")
        print(f"  Cumulative threshold: {self.cumulative_threshold * 100:.0f}%")

        for images, labels in tqdm(data_loader, desc="Extracting activations"):
            batch_activations = []

            for i in range(images.size(0)):
                image = images[i:i+1].to(self.device)  # [1, 3, 224, 224]

                # Forward pass to get activations
                with torch.no_grad():
                    _ = self.model(image)
                    activations = self.activations.clone()  # [1, 1024, 14, 14]

                # Get GradCAM-based channel mask (requires gradients)
                channel_mask, num_selected = self._select_channels_with_gradcam(image)
                channel_selection_stats.append(num_selected)

                # Apply mask: zero-out unselected channels
                # channel_mask: [1024] -> reshape to [1, 1024, 1, 1] for broadcasting
                mask_4d = channel_mask.view(1, 1024, 1, 1).float()
                masked_activations = activations * mask_4d  # [1, 1024, 14, 14]

                batch_activations.append(masked_activations.cpu())

            # Concatenate batch
            all_activations.append(torch.cat(batch_activations, dim=0))

        # Concatenate all batches
        X = torch.cat(all_activations, dim=0)  # [N, 1024, 14, 14]

        # Print statistics
        avg_selected = np.mean(channel_selection_stats)
        std_selected = np.std(channel_selection_stats)
        print(f"\nChannel selection statistics:")
        print(f"  Average channels selected: {avg_selected:.1f} ± {std_selected:.1f} (out of 1024)")
        print(f"  Min: {min(channel_selection_stats)}, Max: {max(channel_selection_stats)}")
        print(f"  Sparsity: {(1024 - avg_selected) / 1024 * 100:.1f}% channels zeroed out")

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

                # Skip channels that are all zeros (not selected by GradCAM)
                if channel_data.abs().sum() < 1e-8:
                    continue

                # 99th percentile clipping
                flat = channel_data.flatten()
                # Only compute quantile on non-zero values
                non_zero_flat = flat[flat > 1e-8]
                if len(non_zero_flat) > 0:
                    scale_factor = torch.quantile(non_zero_flat, 0.99)

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
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Multi-Channel ConvSAE Training (ResNet50 - Two-Level Sparsity)',
                 fontsize=14, fontweight='bold')

    # Row 1: Main losses
    # Reconstruction loss
    axs[0, 0].plot(logs["recon_loss"], color='blue', linewidth=1.5)
    axs[0, 0].set_title("Reconstruction Loss")
    axs[0, 0].set_ylabel("MSE")
    axs[0, 0].set_xlabel("Batch")
    axs[0, 0].grid(True, alpha=0.3)

    # L1 sparsity loss
    axs[0, 1].plot(logs["l1_loss"], color='green', linewidth=1.5)
    axs[0, 1].set_title("L1 Sparsity Loss (Feature Activation)")
    axs[0, 1].set_ylabel("L1")
    axs[0, 1].set_xlabel("Batch")
    axs[0, 1].grid(True, alpha=0.3)

    # Channel sparsity loss
    axs[0, 2].plot(logs["channel_sparsity_loss"], color='purple', linewidth=1.5)
    axs[0, 2].set_title("Channel Sparsity Loss (Encoder Weights)")
    axs[0, 2].set_ylabel("L1 per feature")
    axs[0, 2].set_xlabel("Batch")
    axs[0, 2].grid(True, alpha=0.3)

    # Row 2: Regularization losses
    # Lateral inhibition loss
    axs[1, 0].plot(logs["lateral_loss"], color='orange', linewidth=1.5)
    axs[1, 0].set_title("Lateral Inhibition Loss")
    axs[1, 0].set_ylabel("Correlation")
    axs[1, 0].set_xlabel("Batch")
    axs[1, 0].grid(True, alpha=0.3)

    # Spatial compactness loss
    axs[1, 1].plot(logs["compact_loss"], color='red', linewidth=1.5)
    axs[1, 1].set_title("Spatial Compactness Loss (TV)")
    axs[1, 1].set_ylabel("Total Variation")
    axs[1, 1].set_xlabel("Batch")
    axs[1, 1].grid(True, alpha=0.3)

    # Active neurons percentage (channel-level sparsity)
    axs[1, 2].plot(logs["active_pct"], color='teal', linewidth=1.5)
    axs[1, 2].set_title("Active Channels % (Top-K Channel Selection)")
    axs[1, 2].set_ylabel("Percent (%)")
    axs[1, 2].set_xlabel("Batch")
    axs[1, 2].set_ylim(0, 10)
    axs[1, 2].grid(True, alpha=0.3)

    # Row 3: Summary metrics
    # Total loss
    axs[2, 0].plot(logs["total_loss"], color='black', linewidth=2)
    axs[2, 0].set_title("Total Loss")
    axs[2, 0].set_ylabel("Loss")
    axs[2, 0].set_xlabel("Batch")
    axs[2, 0].grid(True, alpha=0.3)

    # Loss components (log scale)
    axs[2, 1].plot(logs["recon_loss"], label='Recon', alpha=0.7)
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

    # Reconstruction vs Channel Sparsity trade-off
    axs[2, 2].scatter(logs["channel_sparsity_loss"], logs["recon_loss"],
                     c=range(len(logs["recon_loss"])), cmap='viridis',
                     alpha=0.5, s=5)
    axs[2, 2].set_title("Reconstruction vs Channel Sparsity")
    axs[2, 2].set_xlabel("Channel Sparsity Loss")
    axs[2, 2].set_ylabel("Reconstruction Loss")
    axs[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training logs saved to {save_path}")
    plt.close()


def visualize_learned_features(model: MultiChannelConvSAE,
                               num_features: int = 32,
                               save_path: str = 'multichannel_csae_features.png'):
    """Visualize learned decoder features."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Multi-Channel ConvSAE Learned Features (ResNet50)', fontsize=14, fontweight='bold')

    # Get decoder weights [in_channels=1024, hidden_dim, kernel_size, kernel_size]
    decoder_weights = model.decoder.weight.detach().cpu()

    # For 1×1 conv, shape is [1024, hidden_dim, 1, 1]
    # Each feature is a 1024-dimensional vector
    decoder_weights = decoder_weights.squeeze()  # [1024, hidden_dim]

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
    channel_importance = decoder_weights.abs().sum(dim=1).numpy()  # [1024]
    axes[1, 0].bar(range(1024), channel_importance, color='orange', alpha=0.7)
    axes[1, 0].set_title('Input Channel Importance (Sum of Absolute Weights)')
    axes[1, 0].set_xlabel('Input Channel Index (0-1023)')
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
    print("Multi-Channel ConvSAE Training with ResNet50 Backbone")
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
    extractor = ResNet50ActivationExtractor(device=device)

    # Collect activation maps (ALL 1024 channels)
    print("\nExtracting ALL 1024 channels from ResNet50 layer3...")
    X = extractor.collect_activation_maps(
        data_loader,
        normalize=True
    )

    print(f"\nCollected activation maps:")
    print(f"  Shape: {X.shape}")  # [N, 1024, 14, 14]
    print(f"  Device: {X.device}")

    # ========================================
    # 3. SETUP CONVSAE TRAINING
    # ========================================
    print("\n" + "="*80)
    print("Setting up Multi-Channel ConvSAE training...")
    print("="*80)

    # Hyperparameters
    BATCH_SIZE = 64
    INPUT_CHANNELS = 1024   # All ResNet50 layer3 channels
    HIDDEN_DIM = 8192       # Sparse feature dimension (8× expansion)
    KERNEL_SIZE = 1         # 1×1 conv for channel-wise features
    TOP_K = 32             # Number of active features per spatial position

    # Loss weights
    LAMBDA_L1 = 1          # L1 sparsity penalty
    LAMBDA_LAT = 0.05        # Lateral inhibition penalty
    LAMBDA_COMPACT = 0.05   # Spatial compactness penalty
    LAMBDA_CHANNEL_SPARSITY = 0.0  # Feature-channel sparsity penalty

    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    EPOCHS = 20

    print(f"\nTraining Configuration:")
    print(f"  Backbone: ResNet50 (pretrained ImageNet)")
    print(f"  Target layer: layer3 (1024 channels, 14×14 resolution)")
    print(f"  Input Channels: {INPUT_CHANNELS}")
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
        "channel_sparsity_loss": [],
        "active_pct": []
    }

    # ========================================
    # 4. TRAINING LOOP
    # ========================================
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80)
    print("\nExpected Behavior (Two-Level Sparsity):")
    print(f"  • Reconstruction Loss: Decrease to <0.02")
    print(f"  • Channel Sparsity (Level 1): Only top-{TOP_K} channels active per sample")
    print(f"    - Active Neurons: ~{TOP_K/HIDDEN_DIM*100:.2f}% (hard channel selection)")
    print(f"    - Ranking: Based on sum of spatial activations (H×W)")
    print(f"  • Spatial Sparsity (Level 2): L1 regularization within selected channels")
    print(f"    - L1 Loss: Should stabilize (encourages sparse spatial patterns)")
    print(f"    - Spatial Compactness: Decrease (more localized activations)")
    print(f"  • Channel Sparsity: Each feature uses ~50-150 input channels")
    print(f"  • Feature Maps: Sparse at both levels (few channels × sparse spatial patterns)")
    print(f"  • GradCAM: ~150-400 channels selected per image (80% cumulative score)")
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
                      f"Loss: {loss.item():.4f} | Recon: {loss_recon.item():.4f} | "
                      f"Ch-Sp: {loss_channel_sparsity.item():.4f} | "
                      f"Active: {active_pct:.1f}%", end="")

        # Epoch summary
        avg_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}

        print(f"\n[Epoch {epoch+1}/{EPOCHS}] Summary:")
        print(f"  Total Loss: {avg_metrics['total_loss']:.4f}")
        print(f"  Reconstruction: {avg_metrics['recon_loss']:.4f}")
        print(f"  L1 Sparsity (Spatial): {avg_metrics['l1_loss']:.4f}")
        print(f"  Lateral Inhibition: {avg_metrics['lateral_loss']:.4f}")
        print(f"  Spatial Compactness: {avg_metrics['compact_loss']:.4f}")
        print(f"  Channel Sparsity (Encoder): {avg_metrics['channel_sparsity_loss']:.4f}")
        print(f"  Active Channels: {avg_metrics['active_pct']:.2f}% (Target: {TOP_K/HIDDEN_DIM*100:.1f}%)")

        # Check sparsity target (with top-k, should be close to TOP_K/HIDDEN_DIM)
        expected_pct = TOP_K / HIDDEN_DIM * 100
        if abs(avg_metrics['active_pct'] - expected_pct) > 1.0:
            print(f"  ℹ Info: Active channels {avg_metrics['active_pct']:.2f}% vs expected {expected_pct:.2f}%")

        print("-" * 80)

    print("="*80)
    print("Training Complete!")
    print("="*80)

    # ========================================
    # 5. SAVE MODEL
    # ========================================
    print("\nSaving models...")

    torch.save(csae_model.state_dict(), 'multichannel_csae_resnet50_model.pth')
    print("✓ Model state dict: multichannel_csae_resnet50_model.pth")

    joblib.dump(csae_model.cpu(), 'multichannel_csae_resnet50_model.pkl')
    print("✓ Full model: multichannel_csae_resnet50_model.pkl")

    training_info = {
        'config': {
            'backbone': 'ResNet50',
            'target_layer': 'layer3',
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
    joblib.dump(training_info, 'multichannel_csae_resnet50_training_info.pkl')
    print("✓ Training info: multichannel_csae_resnet50_training_info.pkl")

    # ========================================
    # 6. VISUALIZATIONS
    # ========================================
    print("\nGenerating visualizations...")

    plot_training_logs(logs, save_path='multichannel_csae_resnet50_logs.png')
    csae_model = csae_model.to(device)
    visualize_learned_features(csae_model, num_features=64,
                               save_path='multichannel_csae_resnet50_features.png')

    # Generate comprehensive ResNet50 SAE visualization
    print("\nGenerating ResNet50 Multi-Channel SAE visualization...")
    visualize_multichannel_sae_r50(
        model=csae_model,
        sample_activations=X,
        num_samples=4,
        num_channels_to_show=8,
        num_features_to_show=16,
        save_path='multichannel_sae_r50_visualization.png'
    )

    print("\n" + "="*80)
    print("✓ All done! Outputs:")
    print("  - multichannel_csae_resnet50_model.pth")
    print("  - multichannel_csae_resnet50_model.pkl")
    print("  - multichannel_csae_resnet50_training_info.pkl")
    print("  - multichannel_csae_resnet50_logs.png")
    print("  - multichannel_csae_resnet50_features.png")
    print("  - multichannel_sae_r50_visualization.png")
    print("="*80)
