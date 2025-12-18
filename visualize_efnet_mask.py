"""
Multi-Channel ConvSAE Visualization Script (EfficientNet-B0 - Masked Loss Variant)

For a given input image:
1. Extract EfficientNet-B0 features[4] activations (ALL ~80 channels, no masking)
2. Use GradCAM to identify top channels with 85% cumulative score (for visualization)
3. Feed ALL activations to trained Multi-Channel ConvSAE (masked loss variant)
4. CSAE applies two-level sparsity on full input
5. Select top-k most activated feature maps
6. Visualize feature maps as 14×14 heatmaps
7. Show which input channels were selected by GradCAM (for reference)

VARIANT: This visualizes models trained with run_efnet_mask.py, which:
- Uses ALL channels as input (no pre-masking)
- Computes reconstruction loss only on GradCAM-selected channels

Usage:
    # Visualize single image
    python visualize_efnet_mask.py --image_path data/imagenette/tench/n01440764_1.JPEG

    # Visualize multiple images from a class
    python visualize_efnet_mask.py --class_name tench --num_images 3

    # Customize parameters
    python visualize_efnet_mask.py --class_name gas_pump --num_images 5 --top_k_features 16
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
from run_efnet_mask import MultiChannelConvSAE
from src.gradcam import GradCAM


class MultiChannelSAEVisualizerEfNetMask:
    """
    Visualizer for Multi-Channel ConvSAE learned features (EfficientNet-B0 masked loss variant).

    VARIANT: For models trained with run_efnet_mask.py:
    - Uses ALL activation channels as input (no pre-masking)
    - Model was trained with reconstruction loss only on GradCAM-selected channels
    - Visualization shows both full input and GradCAM selection for reference

    For each input image:
    1. Extracts EfficientNet-B0 features[4] activations (ALL ~80 channels)
    2. Identifies GradCAM top channels (for visualization reference)
    3. Passes ALL activations through CSAE to get sparse features
       - CSAE applies two-level sparsity on full input
    4. Selects top-k most activated feature maps
    5. Visualizes feature maps as 14×14 heatmaps
    """

    def __init__(self,
                 csae_model_path: str = 'multichannel_csae_efnet_mask_model.pkl',
                 device='cuda',
                 cumulative_threshold=0.85):
        """
        Args:
            csae_model_path: Path to trained Multi-Channel ConvSAE (masked loss variant)
            device: Device to run on
            cumulative_threshold: GradCAM cumulative threshold for visualization (default: 0.85 = 85%)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.cumulative_threshold = cumulative_threshold

        # Load Multi-Channel ConvSAE (masked loss variant)
        print(f"Loading Multi-Channel ConvSAE (Masked Loss) from {csae_model_path}...")
        self.csae_model = joblib.load(csae_model_path).to(self.device)
        self.csae_model.eval()
        print(f"  ✓ Model loaded: {self.csae_model.in_channels}→{self.csae_model.hidden_dim}, top_k={self.csae_model.top_k}")
        print(f"  ✓ Variant: Trained with masked loss (loss computed only on GradCAM channels)")

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

        # GradCAM for channel selection (for visualization reference)
        print("Setting up GradCAM for visualization reference...")
        self.gradcam = GradCAM(self.efnet, self.target_layer)
        print(f"  ✓ GradCAM threshold: {cumulative_threshold * 100:.0f}% cumulative score")
        print(f"  ℹ Note: GradCAM used for visualization only; model processes ALL channels")

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
        Use GradCAM to identify top channels (for visualization reference).

        Note: In masked loss variant, this is for visualization only.
        The model processes ALL channels.

        Args:
            image: [1, 3, 224, 224] - Input image

        Returns:
            channel_mask: [num_channels] - Binary mask (True for selected channels)
            num_selected: Number of selected channels
            channel_weights: [num_channels] - GradCAM importance scores for all channels
        """
        # Compute GradCAM channel weights
        weights, _, pred_class = self.gradcam.forward(image, class_idx=None, verbose=False)

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

            # Skip channels that are all zeros
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
        Extract top-k activated features for an input image.

        VARIANT: Processes ALL activation channels (no pre-masking).
        GradCAM is computed for visualization reference only.

        Pipeline:
        1. Extract ALL activation channels from EfficientNet-B0
        2. Compute GradCAM mask (for visualization reference)
        3. Normalize ALL channels
        4. Pass ALL channels through CSAE to get sparse features
        5. Select top-k most activated feature maps

        Args:
            image_path: Path to input image
            top_k: Number of top features to extract (default: 16)

        Returns:
            results: Dictionary containing visualization data
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract layer activations (ALL channels)
        with torch.no_grad():
            _ = self.efnet(image_tensor)
            layer_acts = self.layer_activations.clone()  # [1, num_channels, H, W]

        # Compute GradCAM mask (for visualization reference only)
        channel_mask, num_selected, channel_weights = self._select_channels_with_gradcam(image_tensor)

        print(f"  GradCAM reference: {num_selected} channels (85% cumulative score)")
        print(f"  Top 5 channels: {torch.argsort(channel_weights, descending=True)[:5].tolist()}")
        print(f"  ℹ Model processes ALL {self.num_channels} channels (no masking)")

        # Normalize ALL activations (same as training)
        layer_acts_norm = self._normalize_layer_activations(layer_acts)

        # Pass ALL channels through CSAE encoder
        with torch.no_grad():
            _, sparse_features = self.csae_model(layer_acts_norm, use_topk=True)

        # Compute feature importance
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
        - Row 0: Input image + Info (GradCAM reference + Variant note) + Feature importance bar chart
        - Rows 1+: Top-k feature activation maps (heatmaps), 4 per row
        """
        n_features = len(top_features)
        n_cols = 8
        n_rows = 1 + (n_features + 3) // 4

        fig = plt.figure(figsize=(24, 3.5 * n_rows))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)

        # ===== Row 0: Overview =====
        # Column 0-1: Input image
        ax_img = fig.add_subplot(gs[0, 0:2])
        ax_img.imshow(image)
        ax_img.set_title("Input Image", fontsize=12, fontweight='bold')
        ax_img.axis('off')

        # Column 2-3: Info
        ax_info = fig.add_subplot(gs[0, 2:4])
        ax_info.axis('off')

        # Get top 5 channels
        top_5_indices = torch.argsort(channel_weights, descending=True)[:5].tolist()
        top_5_scores = [channel_weights[i].item() for i in top_5_indices]

        info_text = f"VARIANT: Masked Loss Model\n"
        info_text += f"  • Input: ALL {self.num_channels} channels (no masking)\n"
        info_text += f"  • Trained with loss on top 85% only\n\n"
        info_text += f"GradCAM Reference (visualization only):\n"
        info_text += f"  • Selected: {num_selected_channels}/{self.num_channels} channels\n"
        info_text += f"  • Top 5 channels:\n"
        for idx, score in zip(top_5_indices[:3], top_5_scores[:3]):
            info_text += f"    #{idx}: {score:.4f}\n"

        ax_info.text(0.05, 0.5, info_text, fontsize=9, family='monospace',
                    verticalalignment='center', transform=ax_info.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

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
            row = 1 + i // 4
            col = (i % 4) * 2

            # Feature activation map (heatmap)
            ax_feat = fig.add_subplot(gs[row, col:col+2])
            im = ax_feat.imshow(activation_map.numpy(), cmap='hot', interpolation='bilinear')
            ax_feat.set_title(f"Feature {feat_idx}\nImportance: {importance:.2f}",
                             fontsize=10, fontweight='bold')
            ax_feat.axis('off')

            # Colorbar
            cbar = plt.colorbar(im, ax=ax_feat, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)

        plt.suptitle(f'Multi-Channel ConvSAE (EfficientNet-B0 - Masked Loss): Top-{n_features} Activated Features ({self.spatial_size}×{self.spatial_size})\n' +
                    f'Input: ALL {self.num_channels} channels → CSAE: {self.csae_model.hidden_dim} features (trained with masked loss on top 85%)',
                    fontsize=13, fontweight='bold', y=0.998)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def visualize_multichannel_sae_efnet_mask(
    model: MultiChannelConvSAE,
    sample_activations: torch.Tensor,
    num_samples: int = 4,
    num_channels_to_show: int = 8,
    num_features_to_show: int = 16,
    save_path: str = 'multichannel_sae_efnet_mask_visualization.png'
):
    """
    Comprehensive visualization of Multi-Channel ConvSAE (EfficientNet-B0 - Masked Loss) learned features.

    Args:
        model: Trained MultiChannelConvSAE (masked loss variant)
        sample_activations: [N, num_channels, H, W] - Sample activation maps (full, no masking)
        num_samples: Number of sample images to visualize
        num_channels_to_show: Number of input channels to display per sample
        num_features_to_show: Number of CSAE features to display
        save_path: Path to save visualization
    """
    device = next(model.parameters()).device
    model.eval()

    # Select random samples
    indices = torch.randperm(sample_activations.shape[0])[:num_samples]
    samples = sample_activations[indices].to(device)

    # Get sparse features for samples (ALL channels processed)
    with torch.no_grad():
        _, sparse_features = model(samples, use_topk=True)

    # Compute feature importance
    feature_importance = sparse_features.sum(dim=(0, 2, 3)).cpu()
    top_features_idx = torch.argsort(feature_importance, descending=True)[:num_features_to_show]

    # Get decoder weights
    decoder_weights = model.decoder.weight.detach().cpu().squeeze()

    # Create figure
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(4, num_samples + 2, hspace=0.4, wspace=0.3)

    # ===== Row 0: Sample input activations =====
    for i in range(num_samples):
        ax = fig.add_subplot(gs[0, i])

        sample_acts = samples[i].cpu()
        avg_acts = sample_acts.mean(dim=0)
        im = ax.imshow(avg_acts.numpy(), cmap='viridis', interpolation='bilinear')
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax.set_title(f'Sample {i+1}\n(ALL {sample_acts.shape[0]} channels)', fontsize=10, fontweight='bold')
        ax.axis('off')

    # Row 0, last 2 columns: Channel importance
    ax_channel = fig.add_subplot(gs[0, num_samples:])
    channel_importance = decoder_weights.abs().sum(dim=1).numpy()
    top_channels = np.argsort(channel_importance)[-20:][::-1]

    ax_channel.barh(range(len(top_channels)), channel_importance[top_channels],
                    color='orange', alpha=0.8, edgecolor='darkorange')
    ax_channel.set_yticks(range(len(top_channels)))
    ax_channel.set_yticklabels([f'Ch{c}' for c in top_channels], fontsize=8)
    ax_channel.set_xlabel('Importance', fontsize=10)
    ax_channel.set_title('Top 20 Input Channels\n(by decoder weights)', fontsize=10, fontweight='bold')
    ax_channel.grid(True, alpha=0.3, axis='x')
    ax_channel.invert_yaxis()

    # ===== Rows 1-3: Top CSAE features =====
    features_per_row = num_samples + 2
    for feat_idx, global_feat_idx in enumerate(top_features_idx):
        row = 1 + feat_idx // features_per_row
        col = feat_idx % features_per_row

        if row >= 4:
            break

        ax = fig.add_subplot(gs[row, col])

        feat_maps = sparse_features[:, global_feat_idx, :, :].cpu()
        avg_feat_map = feat_maps.mean(dim=0)

        im = ax.imshow(avg_feat_map.numpy(), cmap='hot', interpolation='bilinear')
        ax.set_title(f'Feature {global_feat_idx.item()}\nImp: {feature_importance[global_feat_idx]:.1f}',
                    fontsize=9, fontweight='bold')
        ax.axis('off')

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)

    plt.suptitle(f'Multi-Channel ConvSAE (EfficientNet-B0 - Masked Loss Variant) Feature Visualization\n' +
                f'Input: ALL {model.in_channels} channels (no masking) → Hidden: {model.hidden_dim} features (trained with masked loss)',
                fontsize=13, fontweight='bold')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comprehensive visualization saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Multi-Channel ConvSAE learned features (EfficientNet-B0 - Masked Loss Variant)'
    )
    parser.add_argument('--csae_model', type=str,
                       default='multichannel_csae_efnet_mask_model.pkl',
                       help='Path to trained Multi-Channel ConvSAE model (masked loss variant)')
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
    parser.add_argument('--output_dir', type=str, default='multichannel_sae_efnet_mask_visualizations',
                       help='Output directory for visualizations')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create visualizer
    print("="*80)
    print("Multi-Channel ConvSAE Feature Visualization (EfficientNet-B0 - Masked Loss Variant)")
    print("="*80 + "\n")

    visualizer = MultiChannelSAEVisualizerEfNetMask(
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
        save_path = output_dir / f"{img_name}_features_efnet_mask.png"

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
