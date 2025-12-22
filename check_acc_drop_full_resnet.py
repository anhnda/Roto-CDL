"""
Accuracy Drop Analysis for ResNet18 Full ImageNet-1k CSAE

This script evaluates the impact of replacing original ResNet18 layer3 activations
with CSAE-reconstructed activations on classification accuracy for full ImageNet-1k.

Methodology:
1. Load pretrained ResNet18 model
2. Load trained ImageNet-1k Multi-Channel ConvSAE (masked loss variant)
3. Sample 5K test images (5 images per class from 1000 classes)
4. For each test image:
   a. Extract original activations at layer3
   b. Pass through CSAE to get reconstruction
   c. Replace original with reconstructed activations
   d. Continue forward pass through rest of ResNet18
   e. Compare predictions with/without reconstruction

Metrics:
- Original Accuracy: Classification accuracy with original activations
- Reconstruction Accuracy: Classification accuracy with CSAE reconstructions
- Accuracy Drop: Difference between original and reconstruction accuracy
- Only Top-1 accuracy is reported (no Top-5)

Usage:
    # Default: Use cached test samples (5 images per class = 5000 total)
    python check_acc_drop_full_resnet.py

    # Force resample test set
    python check_acc_drop_full_resnet.py --force_resample --test_images_per_class 5

    # Use different model
    python check_acc_drop_full_resnet.py --csae_model my_model.pkl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import joblib
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, Tuple, List
import sys
from pathlib import Path
from PIL import Image
import io
from collections import defaultdict

sys.path.append('.')
from run_resnet_mask_full import MultiChannelConvSAE
from full_classes import IMAGENET2012_CLASSES

# Import test sampler
import pandas as pd
import random


# ==========================================
# Test Dataset Loader
# ==========================================

class ImageNet1kTestDataset(Dataset):
    """
    Dataset that loads sampled test images from cached samples.

    Uses the same cached test samples as visualize_testmf_resnet.py.
    """

    def __init__(self, test_metadata_path: Path, transform=None):
        """
        Args:
            test_metadata_path: Path to test_metadata.pkl
            transform: Image transforms
        """
        self.transform = transform

        # Load cached samples
        print(f"Loading cached test samples from {test_metadata_path}...")
        metadata = joblib.load(test_metadata_path)
        self.samples = metadata['samples']  # List of (image_bytes, label)

        print(f"  ✓ Loaded {len(self.samples)} test images")
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


def create_test_samples_if_needed(raw_dir: Path, test_dir: Path,
                                  images_per_class: int, force_resample: bool):
    """
    Create test samples if cache doesn't exist or force_resample is True.

    This is a simplified version of ImageNet1kTestSampler from visualize_testmf_resnet.py.
    """
    metadata_path = test_dir / "test_metadata.pkl"

    if metadata_path.exists() and not force_resample:
        print(f"Test samples already cached at {metadata_path}")
        return metadata_path

    print(f"\n{'='*80}")
    print(f"Creating test sample dataset...")
    print(f"{'='*80}")
    print(f"Sampling {images_per_class} test images per class from validation set...")

    test_dir.mkdir(parents=True, exist_ok=True)

    # Class mappings
    wnid_to_idx = {wnid: idx for idx, wnid in enumerate(IMAGENET2012_CLASSES.keys())}

    # Storage
    class_samples = defaultdict(list)

    # Find validation parquet files
    val_parquet_files = sorted(raw_dir.glob("validation-*.parquet"))

    if len(val_parquet_files) == 0:
        raise FileNotFoundError(f"No validation parquet files found in {raw_dir}")

    print(f"Found {len(val_parquet_files)} validation parquet files")

    # Read and sample
    for parquet_file in tqdm(val_parquet_files, desc="Reading validation parquet files"):
        df = pd.read_parquet(parquet_file)

        for idx, row in df.iterrows():
            label = row['label']

            if len(class_samples[label]) < images_per_class:
                image_bytes = row['image']['bytes']
                class_samples[label].append((image_bytes, label))

        # Check if done
        min_samples = min(len(samples) for samples in class_samples.values()) if class_samples else 0
        if min_samples >= images_per_class and len(class_samples) == 1000:
            print(f"\nCollected {images_per_class} samples for all 1000 classes!")
            break

    # Build final sample list
    samples = []
    for class_idx in range(1000):
        if len(class_samples[class_idx]) >= images_per_class:
            sampled = random.sample(class_samples[class_idx], images_per_class)
            samples.extend(sampled)
        else:
            print(f"WARNING: Class {class_idx} has only {len(class_samples[class_idx])} test samples")
            samples.extend(class_samples[class_idx])

    print(f"\nTotal sampled test images: {len(samples)}")

    # Save
    print(f"Saving test samples to {test_dir}...")
    joblib.dump({
        'samples': samples,
        'images_per_class': images_per_class,
        'num_classes': 1000,
        'wnid_to_idx': wnid_to_idx
    }, metadata_path)
    print(f"✓ Test samples cached!")

    return metadata_path


# ==========================================
# ResNet18 with CSAE Reconstruction
# ==========================================

class ResNet18WithCSAEReconstruction:
    """
    ResNet18 model with CSAE reconstruction at layer3.

    Allows evaluation of classification accuracy when original activations
    are replaced with CSAE reconstructions.
    """

    def __init__(self, resnet_model: nn.Module, csae_model: MultiChannelConvSAE, device='cuda'):
        """
        Args:
            resnet_model: Pretrained ResNet18 model
            csae_model: Trained Multi-Channel ConvSAE (ImageNet-1k full)
            device: Device to run on
        """
        self.resnet = resnet_model.to(device)
        self.csae = csae_model.to(device)
        self.device = device

        self.resnet.eval()
        self.csae.eval()

        # Hook to capture activations at layer3
        self.target_layer = self.resnet.layer3
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output

    def _normalize_activations(self, acts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize activations using per-channel 99th percentile.
        Same normalization as used during CSAE training.

        Args:
            acts: [B, C, H, W] - Raw activations

        Returns:
            normalized: [B, C, H, W] - Normalized activations
            scale_factors: [B, C] - Scale factors used for normalization
        """
        B, C, H, W = acts.shape
        normalized = acts.clone()
        scale_factors = torch.ones(B, C, device=acts.device)

        for b in range(B):
            for c in range(C):
                channel_data = acts[b, c, :, :]

                # Skip channels that are all zeros
                if channel_data.abs().sum() < 1e-8:
                    continue

                # 99th percentile normalization
                non_zero_vals = channel_data[channel_data > 1e-8]
                if len(non_zero_vals) > 0:
                    scale_factor = torch.quantile(non_zero_vals, 0.99)

                    if scale_factor > 1e-8:
                        channel_data = torch.clamp(channel_data, min=0.0, max=scale_factor)
                        normalized[b, c, :, :] = channel_data / (scale_factor + 1e-8)
                        scale_factors[b, c] = scale_factor

        return normalized, scale_factors

    def forward_original(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with original activations (no reconstruction).

        Args:
            x: [B, 3, 224, 224] - Input images

        Returns:
            logits: [B, 1000] - Classification logits
        """
        with torch.no_grad():
            return self.resnet(x)

    def forward_with_reconstruction(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with CSAE-reconstructed activations at layer3.

        Args:
            x: [B, 3, 224, 224] - Input images

        Returns:
            logits: [B, 1000] - Classification logits with reconstructed activations
            stats: Dictionary with reconstruction statistics
        """
        with torch.no_grad():
            # Forward through ResNet18 up to layer3
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)
            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)

            # Capture original activations
            original_acts = x.clone()  # [B, C, H, W]

            # Normalize activations (same as training)
            normalized_acts, scale_factors = self._normalize_activations(original_acts)

            # Pass through CSAE to get reconstruction
            reconstruction, sparse_features = self.csae(normalized_acts, use_topk=True)

            # Denormalize reconstruction
            scale_factors_4d = scale_factors.unsqueeze(2).unsqueeze(3)
            reconstructed_acts = reconstruction * scale_factors_4d

            # Replace original with reconstructed activations
            x = reconstructed_acts

            # Continue forward pass through rest of ResNet18
            x = self.resnet.layer4(x)
            x = self.resnet.avgpool(x)
            x = torch.flatten(x, 1)
            logits = self.resnet.fc(x)

            # Compute reconstruction statistics
            mse = F.mse_loss(reconstructed_acts, original_acts).item()
            relative_error = (reconstructed_acts - original_acts).abs().mean() / (original_acts.abs().mean() + 1e-8)
            relative_error = relative_error.item()
            sparsity = (sparse_features > 0).float().mean().item()

            stats = {
                'mse': mse,
                'relative_error': relative_error,
                'sparsity': sparsity
            }

            return logits, stats


# ==========================================
# Evaluation
# ==========================================

def evaluate_accuracy_drop(
    model: ResNet18WithCSAEReconstruction,
    data_loader: DataLoader,
    device: str = 'cuda'
) -> Dict:
    """
    Evaluate classification accuracy with original vs reconstructed activations.

    Only computes Top-1 accuracy (no Top-5).

    Args:
        model: ResNet18WithCSAEReconstruction model
        data_loader: DataLoader with (image, label) pairs
        device: Device to run on

    Returns:
        results: Dictionary with accuracy metrics
    """
    # Metrics
    total_samples = 0
    correct_original = 0
    correct_reconstructed = 0

    # Reconstruction stats
    mse_list = []
    relative_error_list = []
    sparsity_list = []

    # Per-class accuracy
    per_class_correct_original = defaultdict(int)
    per_class_correct_reconstructed = defaultdict(int)
    per_class_total = defaultdict(int)

    print("\nEvaluating accuracy with original vs reconstructed activations...")
    print("="*80)

    for images, labels in tqdm(data_loader, desc="Processing batches"):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Forward with original activations
        logits_original = model.forward_original(images)
        pred_original = logits_original.argmax(dim=1)

        # Forward with reconstructed activations
        logits_reconstructed, stats = model.forward_with_reconstruction(images)
        pred_reconstructed = logits_reconstructed.argmax(dim=1)

        # Top-1 accuracy
        correct_original += (pred_original == labels).sum().item()
        correct_reconstructed += (pred_reconstructed == labels).sum().item()

        # Per-class accuracy
        for i in range(batch_size):
            label = labels[i].item()
            per_class_total[label] += 1
            if pred_original[i] == labels[i]:
                per_class_correct_original[label] += 1
            if pred_reconstructed[i] == labels[i]:
                per_class_correct_reconstructed[label] += 1

        total_samples += batch_size

        # Collect reconstruction stats
        mse_list.append(stats['mse'])
        relative_error_list.append(stats['relative_error'])
        sparsity_list.append(stats['sparsity'])

    # Compute accuracies
    acc_original = correct_original / total_samples * 100
    acc_reconstructed = correct_reconstructed / total_samples * 100
    acc_drop = acc_original - acc_reconstructed

    # Average reconstruction stats
    avg_mse = np.mean(mse_list)
    avg_relative_error = np.mean(relative_error_list)
    avg_sparsity = np.mean(sparsity_list)

    # Compute per-class accuracy drops
    per_class_acc_drop = {}
    for label in per_class_total.keys():
        if per_class_total[label] > 0:
            acc_orig = per_class_correct_original[label] / per_class_total[label] * 100
            acc_recon = per_class_correct_reconstructed[label] / per_class_total[label] * 100
            per_class_acc_drop[label] = acc_orig - acc_recon

    # Find worst classes
    sorted_classes = sorted(per_class_acc_drop.items(), key=lambda x: x[1], reverse=True)
    worst_classes = sorted_classes[:10] if len(sorted_classes) >= 10 else sorted_classes

    results = {
        'total_samples': total_samples,
        'acc_original': acc_original,
        'acc_reconstructed': acc_reconstructed,
        'acc_drop': acc_drop,
        'avg_mse': avg_mse,
        'avg_relative_error': avg_relative_error,
        'avg_sparsity': avg_sparsity,
        'per_class_total': dict(per_class_total),
        'worst_classes': worst_classes
    }

    return results


def print_results(results: Dict):
    """
    Pretty-print accuracy drop analysis results.

    Args:
        results: Dictionary with accuracy metrics
    """
    print("\n" + "="*80)
    print("ACCURACY DROP ANALYSIS RESULTS (ImageNet-1k Full)")
    print("="*80)

    print(f"\nDataset:")
    print(f"  Total samples: {results['total_samples']}")
    print(f"  Number of classes: {len(results['per_class_total'])}")

    print(f"\nTop-1 Accuracy:")
    print(f"  Original (no reconstruction):     {results['acc_original']:.2f}%")
    print(f"  Reconstructed (CSAE):             {results['acc_reconstructed']:.2f}%")
    print(f"  Accuracy Drop:                    {results['acc_drop']:.2f}%")
    if results['acc_original'] > 0:
        print(f"  Relative Drop:                    {results['acc_drop']/results['acc_original']*100:.2f}%")
    else:
        print(f"  Relative Drop:                    N/A")

    print(f"\nReconstruction Quality:")
    print(f"  Average MSE:                      {results['avg_mse']:.6f}")
    print(f"  Average Relative Error:           {results['avg_relative_error']:.6f}")
    print(f"  Average Sparsity (active %):      {results['avg_sparsity']*100:.2f}%")

    # Show worst classes
    if results['worst_classes']:
        print(f"\nTop 10 Classes with Largest Accuracy Drop:")
        for i, (label, drop) in enumerate(results['worst_classes'], 1):
            class_name = list(IMAGENET2012_CLASSES.values())[label]
            class_name_short = class_name[:50] + "..." if len(class_name) > 50 else class_name
            print(f"  {i:2d}. Class {label:3d} ({class_name_short}): {drop:+.2f}%")

    print("\n" + "="*80)

    # Interpretation
    print("\nInterpretation:")
    if results['acc_drop'] < 1.0:
        print("  ✓ Excellent: CSAE preserves almost all classification information (<1% drop)")
    elif results['acc_drop'] < 3.0:
        print("  ✓ Very Good: CSAE preserves most classification information (<3% drop)")
    elif results['acc_drop'] < 5.0:
        print("  ✓ Good: CSAE preserves significant classification information (<5% drop)")
    elif results['acc_drop'] < 10.0:
        print("  ⚠ Moderate: Notable accuracy drop (5-10%), but reconstruction is still useful")
    else:
        print("  ⚠ High: Significant accuracy drop (>10%), CSAE may be too lossy")

    print("="*80 + "\n")


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate accuracy drop with CSAE-reconstructed activations (ImageNet-1k Full)'
    )
    parser.add_argument('--csae_model', type=str,
                       default='imagenet1k_csae_resnet_mask_model.pkl',
                       help='Path to trained ImageNet-1k ConvSAE model')
    parser.add_argument('--raw_data_dir', type=str,
                       default='/data/imagenet_raw/data',
                       help='Path to raw ImageNet parquet files')
    parser.add_argument('--test_data_dir', type=str,
                       default='/data/imagenet1k_sampletest',
                       help='Path to cached test samples')
    parser.add_argument('--test_images_per_class', type=int, default=5,
                       help='Number of test images per class (default: 5, total: 5000)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--force_resample', action='store_true',
                       help='Force resampling of test images')
    parser.add_argument('--output_file', type=str,
                       default='accuracy_drop_imagenet1k_results.txt',
                       help='Output file to save results')

    args = parser.parse_args()

    print("="*80)
    print("ImageNet-1k Full CSAE - Accuracy Drop Analysis")
    print("="*80)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create/load test samples
    print("\nStep 1: Preparing test dataset...")
    test_metadata_path = create_test_samples_if_needed(
        raw_dir=Path(args.raw_data_dir),
        test_dir=Path(args.test_data_dir),
        images_per_class=args.test_images_per_class,
        force_resample=args.force_resample
    )

    # Load ResNet18 model
    print("\nStep 2: Loading ResNet18 model...")
    resnet_model = models.resnet18(pretrained=True)
    resnet_model.eval()
    print("  ✓ ResNet18 loaded")

    # Load CSAE model
    print(f"\nStep 3: Loading CSAE model from {args.csae_model}...")
    csae_model = joblib.load(args.csae_model)
    csae_model.eval()
    print(f"  ✓ CSAE loaded: {csae_model.in_channels}→{csae_model.hidden_dim}, top_k={csae_model.top_k}")

    # Create combined model
    print("\nStep 4: Creating ResNet18WithCSAEReconstruction...")
    model = ResNet18WithCSAEReconstruction(resnet_model, csae_model, device=device)
    print("  ✓ Model ready")

    # Setup data
    print(f"\nStep 5: Loading test dataset...")
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageNet1kTestDataset(test_metadata_path, transform=data_transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total test images: {len(dataset)}")

    # Evaluate accuracy drop
    print("\nStep 6: Evaluating accuracy drop...")
    results = evaluate_accuracy_drop(model, data_loader, device=device)

    # Print results
    print_results(results)

    # Save results to file
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ImageNet-1k Full CSAE - Accuracy Drop Analysis\n")
        f.write("="*80 + "\n\n")

        f.write(f"Configuration:\n")
        f.write(f"  CSAE Model: {args.csae_model}\n")
        f.write(f"  Test Images: {results['total_samples']}\n")
        f.write(f"  Classes: {len(results['per_class_total'])}\n")
        f.write(f"  Batch Size: {args.batch_size}\n\n")

        f.write(f"Top-1 Accuracy:\n")
        f.write(f"  Original:        {results['acc_original']:.2f}%\n")
        f.write(f"  Reconstructed:   {results['acc_reconstructed']:.2f}%\n")
        f.write(f"  Drop:            {results['acc_drop']:.2f}%\n")
        if results['acc_original'] > 0:
            f.write(f"  Relative Drop:   {results['acc_drop']/results['acc_original']*100:.2f}%\n\n")
        else:
            f.write(f"  Relative Drop:   N/A\n\n")

        f.write(f"Reconstruction Quality:\n")
        f.write(f"  Average MSE:             {results['avg_mse']:.6f}\n")
        f.write(f"  Average Relative Error:  {results['avg_relative_error']:.6f}\n")
        f.write(f"  Average Sparsity:        {results['avg_sparsity']*100:.2f}%\n\n")

        # Worst classes
        if results['worst_classes']:
            f.write(f"Top 10 Classes with Largest Accuracy Drop:\n")
            for i, (label, drop) in enumerate(results['worst_classes'], 1):
                class_name = list(IMAGENET2012_CLASSES.values())[label]
                f.write(f"  {i:2d}. Class {label:3d} ({class_name}): {drop:+.2f}%\n")
            f.write("\n")

        f.write("="*80 + "\n")

    print(f"✓ Results saved to {args.output_file}")
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
