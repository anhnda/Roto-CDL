"""
Multi-Channel ConvSAE Visualization on ImageNet-1k Test Set

This script:
1. Samples test images from ImageNet-1k validation set (from parquet files)
2. Caches sampled test images to /data/imagenet1k_sampletest for reuse
3. Visualizes ConvSAE features on test images using trained model

Usage:
    # First run: Extract and cache test samples, then visualize
    python visualize_testmf_resnet.py --num_samples 10 --top_k_features 16

    # Subsequent runs: Use cached test samples
    python visualize_testmf_resnet.py --num_samples 5 --top_k_features 12

    # Force re-sampling
    python visualize_testmf_resnet.py --force_resample --test_images_per_class 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import joblib
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import io
import pandas as pd
import random
from collections import defaultdict
from tqdm import tqdm

# Import our model class
sys.path.append('.')
from run_resnet_mask_full import MultiChannelConvSAE
from src.gradcam import GradCAM
from full_classes import IMAGENET2012_CLASSES


# ==========================================
# Test Image Sampler
# ==========================================

class ImageNet1kTestSampler:
    """
    Samples test images from ImageNet-1k validation parquet files.
    Caches sampled images to disk for reuse.
    """

    def __init__(self,
                 raw_dir: Path,
                 test_dir: Path,
                 images_per_class: int = 5,
                 force_resample: bool = False):
        """
        Args:
            raw_dir: Path to raw ImageNet parquet files
            test_dir: Path to save sampled test images
            images_per_class: Number of test images to sample per class
            force_resample: Force resampling even if cache exists
        """
        self.raw_dir = raw_dir
        self.test_dir = test_dir
        self.images_per_class = images_per_class

        # Create test directory
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.test_dir / "test_metadata.pkl"

        # Check cache
        if self.metadata_path.exists() and not force_resample:
            print(f"\n{'='*80}")
            print(f"Loading cached test samples from {self.test_dir}")
            print(f"{'='*80}")
            self.load_cached_samples()
        else:
            print(f"\n{'='*80}")
            print(f"Creating new test sample dataset...")
            print(f"{'='*80}")
            self.create_test_samples()

    def create_test_samples(self):
        """Sample test images from validation parquet files."""
        print(f"Sampling {self.images_per_class} test images per class...")
        print(f"Total target: {self.images_per_class * 1000} images")

        # Class mappings
        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(IMAGENET2012_CLASSES.keys())}

        # Storage
        class_samples = defaultdict(list)

        # Find validation parquet files
        val_parquet_files = sorted(self.raw_dir.glob("validation-*.parquet"))

        if len(val_parquet_files) == 0:
            raise FileNotFoundError(f"No validation parquet files found in {self.raw_dir}")

        print(f"Found {len(val_parquet_files)} validation parquet files")

        # Read and sample
        for parquet_file in tqdm(val_parquet_files, desc="Reading validation parquet files"):
            df = pd.read_parquet(parquet_file)

            for idx, row in df.iterrows():
                label = row['label']

                if len(class_samples[label]) < self.images_per_class:
                    image_bytes = row['image']['bytes']
                    class_samples[label].append((image_bytes, label))

            # Check if done
            min_samples = min(len(samples) for samples in class_samples.values()) if class_samples else 0
            if min_samples >= self.images_per_class and len(class_samples) == 1000:
                print(f"\nCollected {self.images_per_class} samples for all 1000 classes!")
                break

        # Build final sample list
        self.samples = []
        for class_idx in range(1000):
            if len(class_samples[class_idx]) >= self.images_per_class:
                sampled = random.sample(class_samples[class_idx], self.images_per_class)
                self.samples.extend(sampled)
            else:
                print(f"WARNING: Class {class_idx} has only {len(class_samples[class_idx])} test samples")
                self.samples.extend(class_samples[class_idx])

        print(f"\nTotal sampled test images: {len(self.samples)}")

        # Save
        print(f"Saving test samples to {self.test_dir}...")
        joblib.dump({
            'samples': self.samples,
            'images_per_class': self.images_per_class,
            'num_classes': 1000,
            'wnid_to_idx': self.wnid_to_idx
        }, self.metadata_path)
        print(f"✓ Test samples cached!")

    def load_cached_samples(self):
        """Load cached test samples."""
        metadata = joblib.load(self.metadata_path)
        self.samples = metadata['samples']
        self.wnid_to_idx = metadata['wnid_to_idx']

        print(f"Loaded {len(self.samples)} test images from cache")
        print(f"  Images per class: {metadata['images_per_class']}")
        print(f"  Number of classes: {metadata['num_classes']}")

    def get_random_samples(self, n: int) -> List[Tuple[Image.Image, int]]:
        """
        Get n random test samples.

        Returns:
            List of (PIL Image, label) tuples
        """
        sampled_indices = random.sample(range(len(self.samples)), min(n, len(self.samples)))

        result = []
        for idx in sampled_indices:
            image_bytes, label = self.samples[idx]
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            result.append((image, label))

        return result


# ==========================================
# Multi-Channel SAE Visualizer
# ==========================================

class MultiChannelSAEVisualizerTestMF:
    """
    Visualizer for Multi-Channel ConvSAE on ImageNet-1k test set.
    """

    def __init__(self,
                 csae_model_path: str = 'imagenet1k_csae_resnet_mask_model.pkl',
                 device='cuda',
                 cumulative_threshold=0.85):
        """
        Args:
            csae_model_path: Path to trained Multi-Channel ConvSAE
            device: Device to run on
            cumulative_threshold: GradCAM threshold for visualization
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.cumulative_threshold = cumulative_threshold

        # Load ConvSAE
        print(f"Loading Multi-Channel ConvSAE from {csae_model_path}...")
        self.csae_model = joblib.load(csae_model_path).to(self.device)
        self.csae_model.eval()
        print(f"  ✓ Model loaded: {self.csae_model.in_channels}→{self.csae_model.hidden_dim}, top_k={self.csae_model.top_k}")

        # Load ResNet18
        print("Loading ResNet18 backbone...")
        self.resnet = models.resnet18(pretrained=True).to(self.device)
        self.resnet.eval()
        self.target_layer = self.resnet.layer3

        # Get dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            x = self.resnet.conv1(dummy_input)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)
            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)
            self.num_channels = x.shape[1]
            self.spatial_size = x.shape[2]

        print(f"  ✓ Target layer: layer3, {self.num_channels} channels, {self.spatial_size}×{self.spatial_size}")

        # Hook for activations
        self.layer_activations = None
        self.target_layer.register_forward_hook(self._save_layer_activation)

        # GradCAM
        self.gradcam = GradCAM(self.resnet, self.target_layer)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("✓ Visualizer ready!\n")

    def _save_layer_activation(self, module, input, output):
        """Hook to save layer activations."""
        self.layer_activations = output.detach()

    def _normalize_layer_activations(self, acts: torch.Tensor) -> torch.Tensor:
        """Normalize activations using per-channel 99th percentile."""
        normalized = acts.clone()

        for c in range(acts.shape[1]):
            channel_data = acts[0, c, :, :]

            if channel_data.abs().sum() < 1e-8:
                continue

            non_zero_vals = channel_data[channel_data > 1e-8]
            if len(non_zero_vals) > 0:
                scale_factor = torch.quantile(non_zero_vals, 0.99)

                if scale_factor > 1e-8:
                    channel_data = torch.clamp(channel_data, min=0.0, max=scale_factor)
                    normalized[0, c, :, :] = channel_data / (scale_factor + 1e-8)

        return normalized

    def _select_channels_with_gradcam(self, image: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor, int]:
        """Use GradCAM to identify top channels."""
        weights, _, pred_class = self.gradcam.forward(image, class_idx=None, verbose=False)

        sorted_indices = torch.argsort(weights, descending=True)
        sorted_weights = weights[sorted_indices]

        total_score = sorted_weights.sum()
        if total_score > 0:
            cumsum = torch.cumsum(sorted_weights / total_score, dim=0)
            num_selected = (cumsum < self.cumulative_threshold).sum().item() + 1
            num_selected = min(num_selected, len(sorted_indices))
        else:
            num_selected = max(1, int(0.1 * len(sorted_indices)))

        channel_mask = torch.zeros(self.num_channels, dtype=torch.bool, device=self.device)
        selected_channels = sorted_indices[:num_selected]
        channel_mask[selected_channels] = True

        return channel_mask, num_selected, weights, pred_class

    def extract_features(self, image: Image.Image, label: int, top_k: int = 16) -> Dict:
        """
        Extract top-k activated features for a test image.

        Args:
            image: PIL Image
            label: Ground truth label
            top_k: Number of top features to extract

        Returns:
            Dictionary with visualization data
        """
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract activations
        with torch.no_grad():
            logits = self.resnet(image_tensor)
            layer_acts = self.layer_activations.clone()
            pred_label = logits.argmax(dim=1).item()

        # GradCAM
        channel_mask, num_selected, channel_weights, gradcam_pred = self._select_channels_with_gradcam(image_tensor)

        # Normalize
        layer_acts_norm = self._normalize_layer_activations(layer_acts)

        # CSAE
        with torch.no_grad():
            _, sparse_features = self.csae_model(layer_acts_norm, use_topk=True)

        # Feature importance
        feature_importance = sparse_features.sum(dim=(2, 3)).squeeze()
        top_k_values, top_k_indices = torch.topk(feature_importance, k=min(top_k, len(feature_importance)))

        # Top features
        top_features = []
        for idx, importance in zip(top_k_indices, top_k_values):
            activation_map = sparse_features[0, idx, :, :].cpu()
            top_features.append((idx.item(), importance.item(), activation_map))

        # Get class names
        idx_to_wnid = {idx: wnid for wnid, idx in enumerate(IMAGENET2012_CLASSES.keys())}
        true_class = list(IMAGENET2012_CLASSES.values())[label]
        pred_class = list(IMAGENET2012_CLASSES.values())[pred_label]

        results = {
            'image': image,
            'label': label,
            'pred_label': pred_label,
            'true_class': true_class,
            'pred_class': pred_class,
            'correct': (label == pred_label),
            'num_selected_channels': num_selected,
            'channel_weights': channel_weights.cpu(),
            'top_features': top_features,
            'feature_importance': feature_importance.cpu()
        }

        return results

    def visualize_features(self, image: Image.Image, label: int,
                          top_k: int = 16, save_path: str = None):
        """
        Visualize top-k CSAE features for a test image.

        Args:
            image: PIL Image
            label: Ground truth label
            top_k: Number of features to visualize
            save_path: Path to save visualization
        """
        print(f"Processing test image (label={label})...")

        # Extract features
        results = self.extract_features(image, label, top_k=top_k)

        # Visualize
        print("Generating visualization...")
        self._plot_feature_grid(results, save_path)

        print(f"✓ Visualization complete!")
        if save_path:
            print(f"  Saved to: {save_path}")

    def _plot_feature_grid(self, results: Dict, save_path: str = None):
        """Plot grid of CSAE features for test image."""
        image = results['image']
        top_features = results['top_features']
        true_class = results['true_class']
        pred_class = results['pred_class']
        correct = results['correct']
        num_selected = results['num_selected_channels']

        n_features = len(top_features)
        n_cols = 8
        n_rows = 1 + (n_features + 3) // 4

        fig = plt.figure(figsize=(24, 3.5 * n_rows))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)

        # ===== Row 0: Overview =====
        # Input image
        ax_img = fig.add_subplot(gs[0, 0:2])
        ax_img.imshow(image)
        ax_img.set_title("Test Image", fontsize=12, fontweight='bold')
        ax_img.axis('off')

        # Prediction info
        ax_info = fig.add_subplot(gs[0, 2:4])
        ax_info.axis('off')

        status = "✓ CORRECT" if correct else "✗ WRONG"
        status_color = "green" if correct else "red"

        info_text = f"Prediction: {status}\n"
        info_text += f"  True: {true_class[:40]}...\n" if len(true_class) > 40 else f"  True: {true_class}\n"
        info_text += f"  Pred: {pred_class[:40]}...\n\n" if len(pred_class) > 40 else f"  Pred: {pred_class}\n\n"
        info_text += f"Model: ImageNet-1k Full (1000 classes)\n"
        info_text += f"  • {self.num_channels} input channels\n"
        info_text += f"  • {self.csae_model.hidden_dim} CSAE features\n"
        info_text += f"  • Top-k: {self.csae_model.top_k}\n"
        info_text += f"  • GradCAM: {num_selected}/{self.num_channels} channels"

        ax_info.text(0.05, 0.5, info_text, fontsize=9, family='monospace',
                    verticalalignment='center', transform=ax_info.transAxes,
                    bbox=dict(boxstyle='round',
                             facecolor='lightgreen' if correct else 'lightcoral',
                             alpha=0.3))

        # Feature importance bar chart
        ax_bar = fig.add_subplot(gs[0, 4:])
        importances = [imp for _, imp, _ in top_features]
        feature_indices = [f"F{idx}" for idx, _, _ in top_features]
        ax_bar.bar(range(len(importances)), importances,
                  color='steelblue', alpha=0.8, edgecolor='navy')
        ax_bar.set_xlabel('Feature Index', fontsize=10)
        ax_bar.set_ylabel('Importance', fontsize=10)
        ax_bar.set_title(f'Top-{n_features} CSAE Feature Importance',
                        fontsize=12, fontweight='bold')
        ax_bar.set_xticks(range(len(importances)))
        ax_bar.set_xticklabels(feature_indices, rotation=45, ha='right', fontsize=8)
        ax_bar.grid(True, alpha=0.3, axis='y')

        # ===== Rows 1+: Feature maps =====
        for i, (feat_idx, importance, activation_map) in enumerate(top_features):
            row = 1 + i // 4
            col = (i % 4) * 2

            ax_feat = fig.add_subplot(gs[row, col:col+2])
            im = ax_feat.imshow(activation_map.numpy(), cmap='hot', interpolation='bilinear')
            ax_feat.set_title(f"Feature {feat_idx}\nImp: {importance:.2f}",
                             fontsize=10, fontweight='bold')
            ax_feat.axis('off')

            cbar = plt.colorbar(im, ax=ax_feat, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)

        status_str = "CORRECT" if correct else "WRONG"
        plt.suptitle(f'ImageNet-1k Test Set Visualization ({status_str})\n' +
                    f'Multi-Channel ConvSAE: {self.num_channels} channels → {self.csae_model.hidden_dim} features',
                    fontsize=13, fontweight='bold', y=0.998)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize Multi-Channel ConvSAE on ImageNet-1k Test Set'
    )
    parser.add_argument('--csae_model', type=str,
                       default='imagenet1k_csae_resnet_mask_model.pkl',
                       help='Path to trained ConvSAE model')
    parser.add_argument('--raw_data_dir', type=str,
                       default='/data/imagenet_raw/data',
                       help='Path to raw ImageNet parquet files')
    parser.add_argument('--test_data_dir', type=str,
                       default='/data/imagenet1k_sampletest',
                       help='Path to cached test samples')
    parser.add_argument('--test_images_per_class', type=int, default=5,
                       help='Number of test images to sample per class')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of random test images to visualize')
    parser.add_argument('--top_k_features', type=int, default=16,
                       help='Number of top features to visualize per image')
    parser.add_argument('--output_dir', type=str,
                       default='imagenet1k_test_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--force_resample', action='store_true',
                       help='Force resampling of test images')

    args = parser.parse_args()

    print("="*80)
    print("Multi-Channel ConvSAE Test Set Visualization (ImageNet-1k Full)")
    print("="*80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Sample test images
    print("\nStep 1: Sampling test images...")
    sampler = ImageNet1kTestSampler(
        raw_dir=Path(args.raw_data_dir),
        test_dir=Path(args.test_data_dir),
        images_per_class=args.test_images_per_class,
        force_resample=args.force_resample
    )

    # Get random test samples
    print(f"\nStep 2: Selecting {args.num_samples} random test images...")
    test_samples = sampler.get_random_samples(args.num_samples)
    print(f"Selected {len(test_samples)} test images")

    # Create visualizer
    print("\nStep 3: Loading ConvSAE model...")
    visualizer = MultiChannelSAEVisualizerTestMF(
        csae_model_path=args.csae_model,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Visualize each test image
    print(f"\nStep 4: Generating visualizations...")
    correct_count = 0

    for i, (image, label) in enumerate(test_samples):
        print(f"\n{'='*80}")
        print(f"Test Image {i+1}/{len(test_samples)} (label={label})")
        print(f"{'='*80}\n")

        save_path = output_dir / f"test_sample_{i+1}_label{label}.png"

        # Extract features to check prediction
        results = visualizer.extract_features(image, label, top_k=args.top_k_features)
        if results['correct']:
            correct_count += 1

        # Visualize
        visualizer.visualize_features(
            image, label,
            top_k=args.top_k_features,
            save_path=str(save_path)
        )

    # Summary
    accuracy = (correct_count / len(test_samples)) * 100
    print(f"\n{'='*80}")
    print(f"✓ All visualizations complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Accuracy on visualized samples: {correct_count}/{len(test_samples)} ({accuracy:.1f}%)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
