"""
Multi-Channel ConvSAE Training Script with ResNet18 Backbone on Full ImageNet-1k
(Masked Loss Variant)

This script extends run_resnet_mask.py to work with the FULL ImageNet-1k dataset (1000 classes).

Key Features:
- Loads ImageNet-1k from Hugging Face parquet files
- Samples 50 images per class (50,000 total images) for efficient training
- Caches sampled dataset to /data/imagenet1k_sampled for reuse
- Memory-efficient activation collection in batches
- Same masked loss architecture as run_resnet_mask.py

Memory Management:
- 16 GB VRAM, 64 GB RAM
- Processes activations in chunks to avoid OOM
- Saves/loads intermediate results

Usage:
    python run_resnet_mask_full.py
"""

import torch
torch.cuda.init()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.models as models
from torchvision import transforms
import joblib
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import sys
import os
import hashlib
from pathlib import Path
from PIL import Image
import io
import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict
import random

sys.path.append('.')
from src.gradcam import GradCAM
from full_classes import IMAGENET2012_CLASSES

# ==========================================
# Configuration
# ==========================================

# Paths
IMAGENET_RAW_DIR = Path("/data/imagenet_raw")
IMAGENET_SAMPLED_DIR = Path("/data/imagenet1k_sampled")
CACHE_DIR = Path("cache_activations")

# Sampling parameters
IMAGES_PER_CLASS = 50  # 50 images × 1000 classes = 50,000 images
NUM_CLASSES = 1000

# Memory management
ACTIVATION_BATCH_SIZE = 500  # Process activations in chunks of 500 images
ACTIVATION_CHUNK_SIZE = 100  # Save activation chunks every 100 images

# Training parameters
BATCH_SIZE_COLLECTION = 32  # Batch size for ResNet forward pass
BATCH_SIZE_TRAIN = 64       # Batch size for ConvSAE training

# ==========================================
# Multi-Channel ConvSAE Architecture
# (Same as run_resnet_mask.py)
# ==========================================

class MultiChannelConvSAE(nn.Module):
    """
    Convolutional Sparse Autoencoder for multi-channel input with Two-Level Sparsity.
    """

    def __init__(self, in_channels: int = 256, hidden_dim: int = 2048,
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
            bias=False
        )

        # Initialize encoder with small weights
        nn.init.kaiming_normal_(self.encoder.weight, mode='fan_out', nonlinearity='relu')
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)

        # Initialize decoder weights
        nn.init.kaiming_normal_(self.decoder.weight, mode='fan_in')

    def topk_activation(self, x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """Apply Top-K channel selection."""
        B, C, H, W = x.shape

        # Channel-level sparsity
        channel_importance = x.sum(dim=[2, 3])  # [B, C]
        topk_vals, topk_indices = torch.topk(channel_importance, k=self.top_k, dim=1)

        if threshold > 0:
            threshold_mask = topk_vals > threshold
        else:
            threshold_mask = None

        # Create channel selection mask
        channel_mask = torch.zeros(B, C, device=x.device, dtype=torch.bool)
        channel_mask.scatter_(1, topk_indices, True)

        if threshold_mask is not None:
            for b in range(B):
                valid_channels = topk_indices[b][threshold_mask[b]]
                temp_mask = torch.zeros(C, device=x.device, dtype=torch.bool)
                temp_mask[valid_channels] = True
                channel_mask[b] = temp_mask

        channel_mask_4d = channel_mask.unsqueeze(2).unsqueeze(3)
        result = x * channel_mask_4d.float()

        return result

    def forward(self, x: torch.Tensor, use_topk: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder."""
        features = self.encoder(x)
        features = F.relu(features)

        if use_topk:
            sparse_features = self.topk_activation(features)
        else:
            sparse_features = features

        reconstruction = self.decoder(sparse_features)
        return reconstruction, sparse_features

    def normalize_decoder_weights(self):
        """Normalize decoder weights to have unit norm per feature."""
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
    """Spatial Compactness Regularization using Total Variation."""

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
# ImageNet-1k Dataset Sampler
# ==========================================

class ImageNet1kSampledDataset(Dataset):
    """
    Dataset that loads sampled ImageNet-1k images from parquet files.

    Samples IMAGES_PER_CLASS images per class and caches them for reuse.
    """

    def __init__(self, raw_dir: Path, sampled_dir: Path,
                 images_per_class: int = 50, transform=None, force_resample: bool = False):
        self.raw_dir = raw_dir
        self.sampled_dir = sampled_dir
        self.images_per_class = images_per_class
        self.transform = transform

        # Create sampled directory if needed
        self.sampled_dir.mkdir(parents=True, exist_ok=True)

        # Check if sampled dataset exists
        self.metadata_path = self.sampled_dir / "metadata.pkl"

        if self.metadata_path.exists() and not force_resample:
            print(f"\n{'='*80}")
            print(f"Loading cached sampled dataset from {self.sampled_dir}")
            print(f"{'='*80}")
            self.load_cached_dataset()
        else:
            print(f"\n{'='*80}")
            print(f"Creating new sampled dataset...")
            print(f"{'='*80}")
            self.create_sampled_dataset()

    def create_sampled_dataset(self):
        """Sample images from parquet files and save to disk."""
        print(f"Sampling {self.images_per_class} images per class from {NUM_CLASSES} classes...")
        print(f"Total target images: {self.images_per_class * NUM_CLASSES}")

        # Map class IDs to indices (0-999)
        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(IMAGENET2012_CLASSES.keys())}
        self.idx_to_wnid = {idx: wnid for wnid, idx in self.wnid_to_idx.items()}

        # Storage for sampled images
        class_samples = defaultdict(list)  # class_idx -> [(image_bytes, label)]

        # Find all train parquet files
        train_parquet_files = sorted(self.raw_dir.glob("train-*.parquet"))

        if len(train_parquet_files) == 0:
            raise FileNotFoundError(f"No train parquet files found in {self.raw_dir}")

        print(f"Found {len(train_parquet_files)} train parquet files")

        # Read parquet files and sample images
        for parquet_file in tqdm(train_parquet_files, desc="Reading parquet files"):
            # Read parquet file
            df = pd.read_parquet(parquet_file)

            # Process each row
            for idx, row in df.iterrows():
                # Get label (should be in 'label' column, 0-999)
                label = row['label']

                # Check if we need more samples for this class
                if len(class_samples[label]) < self.images_per_class:
                    # Get image bytes
                    image_bytes = row['image']['bytes']
                    class_samples[label].append((image_bytes, label))

            # Check if we have enough samples for all classes
            min_samples = min(len(samples) for samples in class_samples.values())
            if min_samples >= self.images_per_class and len(class_samples) == NUM_CLASSES:
                print(f"\nCollected {self.images_per_class} samples for all {NUM_CLASSES} classes!")
                break

        # Verify we have enough samples
        print(f"\nSampling complete. Samples per class:")
        for class_idx in range(min(10, NUM_CLASSES)):
            print(f"  Class {class_idx}: {len(class_samples[class_idx])} images")
        print(f"  ...")

        # Randomly sample exactly IMAGES_PER_CLASS per class
        self.samples = []
        for class_idx in range(NUM_CLASSES):
            if len(class_samples[class_idx]) >= self.images_per_class:
                sampled = random.sample(class_samples[class_idx], self.images_per_class)
                self.samples.extend(sampled)
            else:
                print(f"WARNING: Class {class_idx} has only {len(class_samples[class_idx])} samples")
                self.samples.extend(class_samples[class_idx])

        print(f"\nTotal sampled images: {len(self.samples)}")

        # Save sampled dataset
        print(f"Saving sampled dataset to {self.sampled_dir}...")
        joblib.dump({
            'samples': self.samples,
            'images_per_class': self.images_per_class,
            'num_classes': NUM_CLASSES,
            'wnid_to_idx': self.wnid_to_idx,
            'idx_to_wnid': self.idx_to_wnid
        }, self.metadata_path)
        print(f"✓ Sampled dataset cached!")

    def load_cached_dataset(self):
        """Load cached sampled dataset."""
        metadata = joblib.load(self.metadata_path)
        self.samples = metadata['samples']
        self.wnid_to_idx = metadata['wnid_to_idx']
        self.idx_to_wnid = metadata['idx_to_wnid']

        print(f"Loaded {len(self.samples)} images from cache")
        print(f"  Images per class: {metadata['images_per_class']}")
        print(f"  Number of classes: {metadata['num_classes']}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_bytes, label = self.samples[idx]

        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


# ==========================================
# ResNet18 Activation Extractor (NO MASKING)
# ==========================================

class ResNet18ActivationExtractor:
    """
    Extracts activation channels from ResNet18 layer3 WITHOUT masking.
    Also stores GradCAM channel masks for each image.
    """

    def __init__(self, device='cuda', cumulative_threshold=0.85):
        self.device = device
        self.cumulative_threshold = cumulative_threshold

        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True).to(device)
        self.model.eval()

        # Target layer: layer3 (256 channels, 14×14)
        self.target_layer = self.model.layer3

        # Get number of channels
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            x = self.model.conv1(dummy_input)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            self.num_channels = x.shape[1]
            self.spatial_size = x.shape[2]

        print(f"ResNet18 Target Layer: layer3")
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
        """Use GradCAM to select important channels and create a binary mask."""
        # Compute GradCAM channel weights
        weights, _, pred_class = self.gradcam.forward(image, class_idx=class_idx, verbose=False)

        # Sort channels by importance
        sorted_indices = torch.argsort(weights, descending=True)
        sorted_weights = weights[sorted_indices]

        # Cumulative threshold
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

    def collect_activation_maps_chunked(
        self,
        data_loader: DataLoader,
        normalize: bool = True,
        chunk_size: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect activation maps in memory-efficient chunks.

        Returns:
            X: [N, num_channels, H, W] - Full activation maps
            masks: [N, num_channels] - Binary masks for GradCAM-selected channels
        """
        all_activations = []
        all_masks = []
        chunk_activations = []
        chunk_masks = []
        channel_selection_stats = []
        total_processed = 0

        print(f"\nCollecting activation maps from ResNet18 layer3 (chunked processing)...")
        print(f"  Chunk size: {chunk_size} images")
        print(f"  GradCAM threshold: {self.cumulative_threshold * 100:.0f}%")

        for images, labels in tqdm(data_loader, desc="Extracting activations"):
            for i in range(images.size(0)):
                image = images[i:i+1].to(self.device)

                # Forward pass
                with torch.no_grad():
                    _ = self.model(image)
                    activations = self.activations.clone()  # [1, num_channels, H, W]

                # Get GradCAM mask
                channel_mask, num_selected = self._select_channels_with_gradcam(image)
                channel_selection_stats.append(num_selected)

                # Store in chunk
                chunk_activations.append(activations.cpu())
                chunk_masks.append(channel_mask.cpu())
                total_processed += 1

                # Save chunk if needed
                if len(chunk_activations) >= chunk_size:
                    all_activations.append(torch.cat(chunk_activations, dim=0))
                    all_masks.append(torch.stack(chunk_masks, dim=0))

                    print(f"  Processed {total_processed} images...")

                    # Clear chunk
                    chunk_activations = []
                    chunk_masks = []

                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Save remaining chunk
        if len(chunk_activations) > 0:
            all_activations.append(torch.cat(chunk_activations, dim=0))
            all_masks.append(torch.stack(chunk_masks, dim=0))

        # Concatenate all chunks
        print(f"\nConcatenating {len(all_activations)} chunks...")
        X = torch.cat(all_activations, dim=0)
        masks = torch.cat(all_masks, dim=0)

        # Statistics
        avg_selected = np.mean(channel_selection_stats)
        std_selected = np.std(channel_selection_stats)
        print(f"\nChannel selection statistics:")
        print(f"  Average channels selected: {avg_selected:.1f} ± {std_selected:.1f} (out of {self.num_channels})")
        print(f"  Min: {min(channel_selection_stats)}, Max: {max(channel_selection_stats)}")

        print(f"\nCollected {X.shape[0]} activation maps:")
        print(f"  Shape: {X.shape}")
        print(f"  Range: [{X.min():.4f}, {X.max():.4f}]")

        # Robust normalization
        if normalize:
            print("\nApplying robust normalization...")
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

        return X, masks


# ==========================================
# Masked Reconstruction Loss
# ==========================================

def masked_reconstruction_loss(reconstruction: torch.Tensor,
                               target: torch.Tensor,
                               masks: torch.Tensor) -> torch.Tensor:
    """Compute MSE reconstruction loss only on GradCAM-selected channels."""
    masks_4d = masks.unsqueeze(2).unsqueeze(3).float()
    squared_error = (reconstruction - target) ** 2
    masked_squared_error = squared_error * masks_4d

    num_selected = masks.sum(dim=1, keepdim=True).float().clamp(min=1.0)
    loss_per_sample = masked_squared_error.sum(dim=(1, 2, 3)) / (num_selected.squeeze() * reconstruction.shape[2] * reconstruction.shape[3])
    loss = loss_per_sample.mean()

    return loss


# ==========================================
# Visualization Functions
# ==========================================

def plot_training_logs(logs: Dict[str, List], save_path: str = 'imagenet1k_csae_resnet_mask_logs.png'):
    """Plot training metrics."""
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Multi-Channel ConvSAE Training (ResNet18 - ImageNet-1k Full)',
                 fontsize=14, fontweight='bold')

    # Row 1
    axs[0, 0].plot(logs["recon_loss"], color='blue', linewidth=1.5)
    axs[0, 0].set_title("Masked Reconstruction Loss")
    axs[0, 0].set_ylabel("MSE (masked)")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(logs["l1_loss"], color='green', linewidth=1.5)
    axs[0, 1].set_title("L1 Sparsity Loss")
    axs[0, 1].set_ylabel("L1")
    axs[0, 1].grid(True, alpha=0.3)

    axs[0, 2].plot(logs["channel_sparsity_loss"], color='purple', linewidth=1.5)
    axs[0, 2].set_title("Channel Sparsity Loss")
    axs[0, 2].set_ylabel("L1 per feature")
    axs[0, 2].grid(True, alpha=0.3)

    # Row 2
    axs[1, 0].plot(logs["lateral_loss"], color='orange', linewidth=1.5)
    axs[1, 0].set_title("Lateral Inhibition Loss")
    axs[1, 0].set_ylabel("Correlation")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(logs["compact_loss"], color='red', linewidth=1.5)
    axs[1, 1].set_title("Spatial Compactness Loss")
    axs[1, 1].set_ylabel("Total Variation")
    axs[1, 1].grid(True, alpha=0.3)

    axs[1, 2].plot(logs["active_pct"], color='teal', linewidth=1.5)
    axs[1, 2].set_title("Active Channels %")
    axs[1, 2].set_ylabel("Percent (%)")
    axs[1, 2].set_ylim(0, 10)
    axs[1, 2].grid(True, alpha=0.3)

    # Row 3
    axs[2, 0].plot(logs["total_loss"], color='black', linewidth=2)
    axs[2, 0].set_title("Total Loss")
    axs[2, 0].set_ylabel("Loss")
    axs[2, 0].grid(True, alpha=0.3)

    axs[2, 1].plot(logs["recon_loss"], label='Recon', alpha=0.7)
    axs[2, 1].plot(logs["l1_loss"], label='L1', alpha=0.7)
    axs[2, 1].plot(logs["lateral_loss"], label='Lateral', alpha=0.7)
    axs[2, 1].set_title("Loss Components (Log)")
    axs[2, 1].set_yscale('log')
    axs[2, 1].legend(fontsize=7)
    axs[2, 1].grid(True, alpha=0.3)

    axs[2, 2].scatter(logs["channel_sparsity_loss"], logs["recon_loss"],
                     c=range(len(logs["recon_loss"])), cmap='viridis', alpha=0.5, s=5)
    axs[2, 2].set_title("Recon vs Channel Sparsity")
    axs[2, 2].set_xlabel("Channel Sparsity")
    axs[2, 2].set_ylabel("Recon Loss")
    axs[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training logs saved to {save_path}")
    plt.close()


def visualize_learned_features(model: MultiChannelConvSAE, num_features: int = 32,
                               save_path: str = 'imagenet1k_csae_resnet_mask_features.png'):
    """Visualize learned decoder features."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('ConvSAE Features (ImageNet-1k Full)', fontsize=14, fontweight='bold')

    decoder_weights = model.decoder.weight.detach().cpu().squeeze()

    # Weight distribution
    weights_flat = decoder_weights.flatten().numpy()
    axes[0, 0].hist(weights_flat, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(weights_flat), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(weights_flat):.3f}')
    axes[0, 0].set_title('Decoder Weight Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Feature L2 norms
    feature_norms = decoder_weights.norm(dim=0).numpy()
    n_show = min(num_features, len(feature_norms))
    axes[0, 1].bar(range(n_show), feature_norms[:n_show], color='green', alpha=0.7)
    axes[0, 1].set_title(f'Feature Magnitudes (Top {n_show})')
    axes[0, 1].grid(True, alpha=0.3)

    # Channel importance
    channel_importance = decoder_weights.abs().sum(dim=1).numpy()
    axes[1, 0].bar(range(len(channel_importance)), channel_importance, color='orange', alpha=0.7)
    axes[1, 0].set_title('Input Channel Importance')
    axes[1, 0].grid(True, alpha=0.3)

    # Feature sparsity
    feature_sparsity = (decoder_weights.abs() > 1e-3).float().sum(dim=0).numpy()
    axes[1, 1].hist(feature_sparsity, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(feature_sparsity), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(feature_sparsity):.1f}')
    axes[1, 1].set_title('Feature Sparsity')
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
    print("="*80)
    print("Multi-Channel ConvSAE Training on Full ImageNet-1k (1000 classes)")
    print("="*80)
    print(f"\nDataset Configuration:")
    print(f"  Raw data: {IMAGENET_RAW_DIR}")
    print(f"  Sampled data: {IMAGENET_SAMPLED_DIR}")
    print(f"  Images per class: {IMAGES_PER_CLASS}")
    print(f"  Total classes: {NUM_CLASSES}")
    print(f"  Total images: {IMAGES_PER_CLASS * NUM_CLASSES}")

    # ========================================
    # 1. SETUP DATA
    # ========================================
    print(f"\n{'='*80}")
    print("Setting up dataset...")
    print(f"{'='*80}")

    # ImageNet preprocessing
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create sampled dataset
    dataset = ImageNet1kSampledDataset(
        raw_dir=IMAGENET_RAW_DIR,
        sampled_dir=IMAGENET_SAMPLED_DIR,
        images_per_class=IMAGES_PER_CLASS,
        transform=data_transform,
        force_resample=False
    )

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE_COLLECTION, shuffle=False, num_workers=4)

    print(f"\nDataset ready: {len(dataset)} images")

    # ========================================
    # 2. EXTRACT ACTIVATION MAPS
    # ========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create activation extractor
    extractor = ResNet18ActivationExtractor(device=device)

    # Collect activations in chunks
    print(f"\nExtracting activations (memory-efficient chunked processing)...")
    X, masks = extractor.collect_activation_maps_chunked(
        data_loader,
        normalize=True,
        chunk_size=ACTIVATION_CHUNK_SIZE
    )

    print(f"\nActivations collected:")
    print(f"  Shape: {X.shape}")
    print(f"  Masks shape: {masks.shape}")

    # ========================================
    # 3. SETUP CONVSAE TRAINING
    # ========================================
    print(f"\n{'='*80}")
    print("Setting up ConvSAE training...")
    print(f"{'='*80}")

    # Hyperparameters
    INPUT_CHANNELS = extractor.num_channels
    HIDDEN_DIM = INPUT_CHANNELS * 8
    KERNEL_SIZE = 1
    TOP_K = int(HIDDEN_DIM * 0.015)

    LAMBDA_L1 = 3.0
    LAMBDA_LAT = 0.01
    LAMBDA_COMPACT = 0.01
    LAMBDA_CHANNEL_SPARSITY = 0.0

    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    EPOCHS = 15

    print(f"\nTraining Configuration:")
    print(f"  Input Channels: {INPUT_CHANNELS}")
    print(f"  Hidden Dim: {HIDDEN_DIM} ({HIDDEN_DIM/INPUT_CHANNELS:.1f}× expansion)")
    print(f"  Top-K: {TOP_K} ({TOP_K/HIDDEN_DIM*100:.1f}%)")
    print(f"  Learning Rate: {LR}")
    print(f"  Epochs: {EPOCHS}")

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
    train_dataset = TensorDataset(X, masks)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, drop_last=True)

    # Logging
    logs = {
        "total_loss": [], "recon_loss": [], "l1_loss": [],
        "lateral_loss": [], "compact_loss": [], "channel_sparsity_loss": [],
        "active_pct": []
    }

    # ========================================
    # 4. TRAINING LOOP
    # ========================================
    print(f"\n{'='*80}")
    print("Starting Training...")
    print(f"{'='*80}")

    for epoch in range(EPOCHS):
        epoch_metrics = {k: 0 for k in logs.keys()}
        n_batches = 0

        for batch_idx, (batch_acts, batch_masks) in enumerate(train_loader):
            batch_acts = batch_acts.to(device)
            batch_masks = batch_masks.to(device)

            optimizer.zero_grad()

            # Forward pass
            reconstruction, sparse_features = csae_model(batch_acts, use_topk=True)

            # Losses
            loss_recon = masked_reconstruction_loss(reconstruction, batch_acts, batch_masks)
            loss_l1 = sparse_features.abs().mean()
            loss_lateral = lat_inhib_loss(sparse_features)
            loss_compact = compact_loss_fn(sparse_features)
            loss_channel_sparsity = channel_sparsity_loss_fn(csae_model.encoder.weight)

            loss = (loss_recon +
                   LAMBDA_L1 * loss_l1 +
                   LAMBDA_LAT * loss_lateral +
                   LAMBDA_COMPACT * loss_compact +
                   LAMBDA_CHANNEL_SPARSITY * loss_channel_sparsity)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(csae_model.parameters(), max_norm=1.0)
            optimizer.step()

            # Normalize decoder
            csae_model.normalize_decoder_weights()

            # Metrics
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

            if batch_idx % 20 == 0:
                print(f"\rEpoch {epoch+1}/{EPOCHS} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | Recon: {loss_recon.item():.4f} | "
                      f"Active: {active_pct:.1f}%", end="")

        # Epoch summary
        avg_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
        print(f"\n[Epoch {epoch+1}/{EPOCHS}] Summary:")
        print(f"  Total Loss: {avg_metrics['total_loss']:.4f}")
        print(f"  Reconstruction: {avg_metrics['recon_loss']:.4f}")
        print(f"  Active Channels: {avg_metrics['active_pct']:.2f}%")
        print("-" * 80)

    print("="*80)
    print("Training Complete!")
    print("="*80)

    # ========================================
    # 5. SAVE MODEL
    # ========================================
    print("\nSaving models...")

    torch.save(csae_model.state_dict(), 'imagenet1k_csae_resnet_mask_model.pth')
    print("✓ Model state dict: imagenet1k_csae_resnet_mask_model.pth")

    joblib.dump(csae_model.cpu(), 'imagenet1k_csae_resnet_mask_model.pkl')
    print("✓ Full model: imagenet1k_csae_resnet_mask_model.pkl")

    training_info = {
        'config': {
            'backbone': 'ResNet18',
            'dataset': 'ImageNet-1k',
            'num_classes': NUM_CLASSES,
            'images_per_class': IMAGES_PER_CLASS,
            'total_images': len(dataset),
            'input_channels': INPUT_CHANNELS,
            'hidden_dim': HIDDEN_DIM,
            'top_k': TOP_K,
            'lr': LR,
            'epochs': EPOCHS,
        },
        'logs': logs,
        'final_metrics': avg_metrics
    }
    joblib.dump(training_info, 'imagenet1k_csae_resnet_mask_training_info.pkl')
    print("✓ Training info: imagenet1k_csae_resnet_mask_training_info.pkl")

    # ========================================
    # 6. VISUALIZATIONS
    # ========================================
    print("\nGenerating visualizations...")

    plot_training_logs(logs, save_path='imagenet1k_csae_resnet_mask_logs.png')
    csae_model = csae_model.to(device)
    visualize_learned_features(csae_model, num_features=64,
                               save_path='imagenet1k_csae_resnet_mask_features.png')

    print("\n" + "="*80)
    print("✓ All done! Outputs:")
    print("  - imagenet1k_csae_resnet_mask_model.pth")
    print("  - imagenet1k_csae_resnet_mask_model.pkl")
    print("  - imagenet1k_csae_resnet_mask_training_info.pkl")
    print("  - imagenet1k_csae_resnet_mask_logs.png")
    print("  - imagenet1k_csae_resnet_mask_features.png")
    print(f"  - Sampled dataset cached in: {IMAGENET_SAMPLED_DIR}")
    print("="*80)
