"""
Multi-Channel ConvSAE Visualization Script (EfficientNet-B0) - Two-Level Sparsity

For a given input image:
1. Extract EfficientNet-B0 features[4] activations (~80 channels, 14×14)
2. Use GradCAM to select top channels with 85% cumulative score
3. Mask other channels (zero them out)
4. Feed masked activations to trained Multi-Channel ConvSAE with two-level sparsity:
   - Level 1 (Channel): Top-k channel selection based on spatial activation sum
   - Level 2 (Spatial): L1-sparse activations within selected channels
5. Obtain high-dimensional encoding activations (8× expansion features)
6. Select top-k most activated feature maps
7. Visualize these feature maps directly (14×14 heatmaps)

Usage:
    # Visualize single image
    python visualize_multichannel_sae_efnet.py --image_path data/imagenette/tench/n01440764_1.JPEG

    # Visualize multiple images from a class
    python visualize_multichannel_sae_efnet.py --class_name tench --num_images 3

    # Customize parameters
    python visualize_multichannel_sae_efnet.py --class_name gas_pump --num_images 5 --top_k_features 16
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

# Import our model class
sys.path.append('.')
from run_csae_efnet import MultiChannelConvSAE
from src.gradcam import GradCAM


class MultiChannelSAEVisualizerEfNet:
    """
    Visualizer for Multi-Channel ConvSAE learned features (EfficientNet-B0 backbone).
    Uses model with two-level sparsity mechanism.

    For each input image:
    1. Extracts EfficientNet-B0 features[4] activations (~80 channels, 14×14)
    2. Uses GradCAM to select top channels with 85% cumulative score
    3. Masks other channels (zeros them out)
    4. Passes masked activations through CSAE to get sparse features
       - CSAE applies two-level sparsity:
         a) Channel-level: Top-k channel selection based on sum of spatial activations
         b) Spatial-level: L1-sparse activations within selected channels
    5. Selects top-k most activated feature maps
    6. Visualizes feature maps directly as 14×14 heatmaps
    """

    def __init__(self,
                 csae_model_path: str = 'multichannel_csae_efnet_model.pkl',
                 device='cuda',
                 cumulative_threshold=0.85):
        """
        Args:
            csae_model_path: Path to trained Multi-Channel ConvSAE
            device: Device to run on
            cumulative_threshold: GradCAM cumulative threshold for channel selection (default: 0.85 = 85%)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.cumulative_threshold = cumulative_threshold

        # Load Multi-Channel ConvSAE
        print(f"Loading Multi-Channel ConvSAE from {csae_model_path}...")
        self.csae_model = joblib.load(csae_model_path).to(self.device)
        self.csae_model.eval()
        print(f"  ✓ Model loaded: {self.csae_model.in_channels}→{self.csae_model.hidden_dim}, top_k={self.csae_model.top_k}")

        # Load EfficientNet-B0 backbone
        print("Loading EfficientNet-B0 backbone...")
        self.efnet = models.efficientnet_b0(pretrained=True).to(self.device)
        self.efnet.eval()

        # Target layer: features[4] (middle layer)
        self.target_layer = self.efnet.features[4]

        # Get the actual number of channels from target layer
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            x = dummy_input
            for i in range(5):  # features[0] through features[4]
                x = self.efnet.features[i](x)
            self.num_channels = x.shape[1]
            self.spatial_size = x.shape[2]

        print(f"  ✓ Target layer: features[4], {self.num_channels} channels, {self.spatial_size}×{self.spatial_size}")

        # Hook for features[4] activations
        self.layer_activations = None
        self.target_layer.register_forward_hook(self._save_layer_activation)

        # GradCAM for channel selection
        print("Setting up GradCAM for channel selection...")
        self.gradcam = GradCAM(self.efnet, self.target_layer)
        print(f"  ✓ GradCAM threshold: {cumulative_threshold * 100:.0f}% cumulative score")

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

    def _select_channels_with_gradcam(self, image: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Use GradCAM to select top channels with cumulative score ≥ threshold (e.g., 85%).

        Args:
            image: [1, 3, 224, 224] - Input image

        Returns:
            channel_mask: [num_channels] - Binary mask (True for selected channels)
            num_selected: Number of selected channels
            channel_weights: [num_channels] - GradCAM importance scores for all channels
        """
        # Compute GradCAM channel weights
        weights, _, pred_class = self.gradcam.forward(image, class_idx=None, verbose=False)

        # weights: [num_channels] - GradCAM importance scores (already ReLU'd)

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
        channel_mask = torch.zeros(self.num_channels, dtype=torch.bool, device=self.device)
        selected_channels = sorted_indices[:num_selected]
        channel_mask[selected_channels] = True

        return channel_mask, num_selected, weights

    def _normalize_layer_activations(self, acts: torch.Tensor) -> torch.Tensor:
        """
        Normalize layer activations using per-channel 99th percentile.

        Args:
            acts: [1, num_channels, H, W] - Raw layer activations

        Returns:
            normalized: [1, num_channels, H, W] - Normalized activations
        """
        normalized = acts.clone()

        # Normalize each channel independently
        for c in range(acts.shape[1]):
            channel_data = acts[0, c, :, :]

            # Skip channels that are all zeros (not selected by GradCAM)
            if channel_data.abs().sum() < 1e-8:
                continue

            # Only compute quantile on non-zero values
            non_zero_vals = channel_data[channel_data > 1e-8]
            if len(non_zero_vals) > 0:
                scale_factor = torch.quantile(non_zero_vals, 0.99)

                if scale_factor > 1e-8:
                    channel_data = torch.clamp(channel_data, min=0.0, max=scale_factor)
                    normalized[0, c, :, :] = channel_data / (scale_factor + 1e-8)

        return normalized

    def extract_features(self, image_path: str, top_k: int = 16) -> Dict:
        """
        Extract top-k activated features for an input image using GradCAM channel selection
        and two-level sparsity.

        Pipeline:
        1. Use GradCAM to select channels with 85% cumulative score
        2. Mask other channels (zero them out)
        3. Normalize selected channels
        4. Pass through CSAE to get sparse features
           - CSAE encoder applies two-level sparsity:
             a) Channel-level: Ranks features by sum(H×W), keeps top-k channels
             b) Spatial-level: Within selected channels, L1 regularization creates sparse patterns
        5. Select top-k most activated feature maps (ranked by spatial sum)

        Args:
            image_path: Path to input image
            top_k: Number of top features to extract (default: 16)

        Returns:
            results: Dictionary containing:
                - image: Original PIL image
                - image_tensor: Preprocessed image tensor
                - layer_acts: Full layer activations [1, num_channels, H, W]
                - layer_masked: Masked activations [1, num_channels, H, W]
                - num_selected_channels: Number of channels selected by GradCAM
                - channel_mask: Binary mask [num_channels] for selected channels
                - channel_weights: GradCAM weights [num_channels]
                - sparse_features: CSAE sparse features [1, hidden_dim, H, W] (two-level sparse)
                - top_features: List of (feature_idx, importance, activation_map)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract layer activations
        with torch.no_grad():
            _ = self.efnet(image_tensor)
            layer_acts = self.layer_activations.clone()  # [1, num_channels, H, W]

        # Use GradCAM to select channels with 85% cumulative score
        channel_mask, num_selected, channel_weights = self._select_channels_with_gradcam(image_tensor)

        print(f"  GradCAM selected {num_selected} channels (85% cumulative score)")
        print(f"  Top 5 channels: {torch.argsort(channel_weights, descending=True)[:5].tolist()}")

        # Apply mask: zero-out unselected channels
        mask_4d = channel_mask.view(1, self.num_channels, 1, 1).float()
        layer_masked = layer_acts * mask_4d  # [1, num_channels, H, W]

        # Normalize masked activations (same as training)
        layer_masked_norm = self._normalize_layer_activations(layer_masked)

        # Pass through CSAE encoder (with Two-Level Sparsity)
        # Level 1: Top-k channel selection based on sum(H×W) per feature
        # Level 2: L1-sparse spatial activations within selected channels
        with torch.no_grad():
            _, sparse_features = self.csae_model(layer_masked_norm, use_topk=True)

        # Compute feature importance (sum of activations per feature channel)
        # This ranks features by their total spatial activation (same criterion used for channel selection)
        # sparse_features: [1, hidden_dim, H, W]
        feature_importance = sparse_features.sum(dim=(2, 3)).squeeze()  # [hidden_dim]

        # Get top-k features
        top_k_values, top_k_indices = torch.topk(feature_importance, k=min(top_k, len(feature_importance)))

        # Extract activation maps for top features
        top_features = []
        for idx, importance in zip(top_k_indices, top_k_values):
            activation_map = sparse_features[0, idx, :, :].cpu()  # [H, W]
            top_features.append((idx.item(), importance.item(), activation_map))

        print(f"  ✓ Extracted {len(top_features)} top features")

        results = {
            'image': image,
            'image_tensor': image_tensor,
            'layer_acts': layer_acts.cpu(),
            'layer_masked': layer_masked.cpu(),
            'num_selected_channels': num_selected,
            'channel_mask': channel_mask.cpu(),
            'channel_weights': channel_weights.cpu(),
            'sparse_features': sparse_features.cpu(),
            'top_features': top_features,
            'feature_importance': feature_importance.cpu()
        }

        return results


    def visualize_features(self, image_path: str, top_k: int = 16, save_path: str = None):
        """
        Visualize top-k CSAE features as heatmaps.

        Args:
            image_path: Path to input image
            top_k: Number of top features to visualize
            save_path: Path to save visualization
        """
        print(f"Processing image: {image_path}")
        print(f"Extracting top-{top_k} features...")

        # Extract features
        results = self.extract_features(image_path, top_k=top_k)

        image = results['image']
        top_features = results['top_features']
        num_selected_channels = results['num_selected_channels']
        channel_weights = results['channel_weights']

        # Visualize
        print("Generating visualization...")
        self._plot_feature_grid(image, top_features, num_selected_channels, channel_weights, save_path)

        print(f"✓ Visualization complete!")
        if save_path:
            print(f"  Saved to: {save_path}")

    def _plot_feature_grid(self, image: Image.Image, top_features: List[Tuple],
                          num_selected_channels: int, channel_weights: torch.Tensor,
                          save_path: str = None):
        """
        Plot grid of top CSAE feature maps.

        Layout:
        - Row 0: Input image + GradCAM info + Feature importance bar chart
        - Rows 1+: Top-k feature activation maps (heatmaps), 4 per row
        """
        n_features = len(top_features)
        n_cols = 8  # 4 features per row, 2 columns per feature (map + colorbar space)
        n_rows = 1 + (n_features + 3) // 4  # Header row + feature rows

        fig = plt.figure(figsize=(24, 3.5 * n_rows))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)

        # ===== Row 0: Overview =====
        # Column 0-1: Input image
        ax_img = fig.add_subplot(gs[0, 0:2])
        ax_img.imshow(image)
        ax_img.set_title("Input Image", fontsize=12, fontweight='bold')
        ax_img.axis('off')

        # Column 2-3: GradCAM channel info
        ax_info = fig.add_subplot(gs[0, 2:4])
        ax_info.axis('off')

        # Get top 5 channels
        top_5_indices = torch.argsort(channel_weights, descending=True)[:5].tolist()
        top_5_scores = [channel_weights[i].item() for i in top_5_indices]

        info_text = f"GradCAM Channel Selection:\n"
        info_text += f"  • Selected: {num_selected_channels}/{self.num_channels} channels\n"
        info_text += f"  • Threshold: 85% cumulative score\n"
        info_text += f"  • Top 5 channels:\n"
        for idx, score in zip(top_5_indices, top_5_scores):
            info_text += f"    #{idx}: {score:.4f}\n"

        ax_info.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                    verticalalignment='center', transform=ax_info.transAxes)

        # Column 4+: Feature importance bar chart
        ax_bar = fig.add_subplot(gs[0, 4:])
        importances = [imp for _, imp, _ in top_features]
        feature_indices = [f"F{idx}" for idx, _, _ in top_features]
        ax_bar.bar(range(len(importances)), importances, color='steelblue', alpha=0.8, edgecolor='navy')
        ax_bar.set_xlabel('Feature Index', fontsize=10)
        ax_bar.set_ylabel('Importance (sum of activations)', fontsize=10)
        ax_bar.set_title(f'Top-{n_features} CSAE Feature Importance', fontsize=12, fontweight='bold')
        ax_bar.set_xticks(range(len(importances)))
        ax_bar.set_xticklabels(feature_indices, rotation=45, ha='right', fontsize=8)
        ax_bar.grid(True, alpha=0.3, axis='y')

        # ===== Rows 1+: Feature activation maps =====
        for i, (feat_idx, importance, activation_map) in enumerate(top_features):
            row = 1 + i // 4  # 4 features per row
            col = (i % 4) * 2  # Each feature takes 2 columns

            # Feature activation map (heatmap)
            ax_feat = fig.add_subplot(gs[row, col:col+2])
            im = ax_feat.imshow(activation_map.numpy(), cmap='hot', interpolation='bilinear')
            ax_feat.set_title(f"Feature {feat_idx}\nImportance: {importance:.2f}",
                             fontsize=10, fontweight='bold')
            ax_feat.axis('off')

            # Colorbar
            cbar = plt.colorbar(im, ax=ax_feat, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)

        plt.suptitle(f'Multi-Channel ConvSAE (EfficientNet-B0): Top-{n_features} Activated Features ({self.spatial_size}×{self.spatial_size} Heatmaps)\n' +
                    f'GradCAM: {num_selected_channels} channels selected → CSAE: {self.csae_model.hidden_dim} features → Top-{n_features} visualized',
                    fontsize=14, fontweight='bold', y=0.998)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Multi-Channel ConvSAE learned features (EfficientNet-B0)'
    )
    parser.add_argument('--csae_model', type=str,
                       default='multichannel_csae_efnet_model.pkl',
                       help='Path to trained Multi-Channel ConvSAE model')
    parser.add_argument('--image_path', type=str,
                       help='Path to input image')
    parser.add_argument('--class_name', type=str,
                       help='Class name to sample images from (alternative to --image_path)')
    parser.add_argument('--num_images', type=int, default=1,
                       help='Number of images to process (if using --class_name)')
    parser.add_argument('--data_dir', type=str, default='data/imagenette',
                       help='Path to Imagenette dataset')
    parser.add_argument('--top_k_features', type=int, default=16,
                       help='Number of top features to visualize per image')
    parser.add_argument('--output_dir', type=str, default='multichannel_sae_efnet_visualizations',
                       help='Output directory for visualizations')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create visualizer
    print("="*80)
    print("Multi-Channel ConvSAE Feature Visualization (EfficientNet-B0)")
    print("="*80 + "\n")

    visualizer = MultiChannelSAEVisualizerEfNet(
        csae_model_path=args.csae_model,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Get list of images to process
    image_paths = []
    if args.image_path:
        image_paths = [args.image_path]
    elif args.class_name:
        class_dir = Path(args.data_dir) / args.class_name
        if not class_dir.exists():
            print(f"Error: Class directory not found: {class_dir}")
            return
        all_images = list(class_dir.glob('*.JPEG')) + list(class_dir.glob('*.jpg'))
        image_paths = all_images[:args.num_images]
    else:
        print("Error: Must specify either --image_path or --class_name")
        return

    print(f"Processing {len(image_paths)} image(s)...\n")

    # Process each image
    for i, img_path in enumerate(image_paths):
        print(f"\n{'='*80}")
        print(f"Image {i+1}/{len(image_paths)}: {img_path.name if isinstance(img_path, Path) else Path(img_path).name}")
        print(f"{'='*80}\n")

        # Generate save path
        img_name = Path(img_path).stem
        save_path = output_dir / f"{img_name}_features_efnet.png"

        visualizer.visualize_features(
            str(img_path),
            top_k=args.top_k_features,
            save_path=str(save_path)
        )

    print(f"\n{'='*80}")
    print(f"✓ All visualizations complete!")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
