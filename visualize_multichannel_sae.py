"""
Multi-Channel ConvSAE Visualization Script

For a given input image:
1. Extract ResNet18 layer3 activations (256 channels, 14×14)
2. Use GradCAM to identify the top channel (highest scoring)
3. Pass the top channel through trained Multi-Channel ConvSAE with Top-K activation
4. Identify top-k most activated feature maps (e.g., top 16 out of 4096)
5. For each top feature, use DeconvNet/Guided Backpropagation to find
   which input pixels contributed to that feature activation
6. Visualize results: input image + top channel + top features + saliency maps

Usage:
    # Visualize single image
    python visualize_multichannel_sae.py --image_path data/imagenette/tench/n01440764_1.JPEG

    # Visualize multiple images from a class
    python visualize_multichannel_sae.py --class_name tench --num_images 3

    # Customize number of top features
    python visualize_multichannel_sae.py --image_path ... --top_k_features 32
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
from run_multichannel_csae_resnet18 import MultiChannelConvSAE
from src.gradcam import GradCAM


class GuidedBackpropReLU(nn.Module):
    """
    Guided Backpropagation through ReLU.
    Only propagates positive gradients (guided backprop rule).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x)

    def backward(self, grad_output):
        # Only propagate positive gradients
        return torch.clamp(grad_output, min=0)


class DeconvNet:
    """
    DeconvNet for visualizing which input pixels activate specific features.

    Uses guided backpropagation to trace feature activations back to input image.
    """

    def __init__(self, resnet_model: nn.Module, device='cuda'):
        """
        Args:
            resnet_model: ResNet18 model (pretrained)
            device: Device to run on
        """
        self.device = device
        self.model = resnet_model.to(device)
        self.model.eval()

        # Hook for layer3 activations
        self.layer3_activations = None
        self.layer3_gradients = None

        # Register hooks
        self.model.layer3.register_forward_hook(self._save_activation)
        self.model.layer3.register_full_backward_hook(self._save_gradient)

        # Replace ReLU with Guided ReLU for visualization
        self._replace_relu_with_guided()

    def _save_activation(self, module, input, output):
        """Save layer3 activations during forward pass."""
        self.layer3_activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Save layer3 gradients during backward pass."""
        self.layer3_gradients = grad_output[0].detach()

    def _replace_relu_with_guided(self):
        """Replace ReLU with Guided ReLU for guided backprop."""
        # Note: This is a simplified version. Full implementation would recursively
        # replace all ReLU layers in the model
        pass

    def compute_saliency_map(self, image: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Compute saliency map showing which input pixels contribute to a feature.

        Args:
            image: [1, 3, 224, 224] - Input image (requires grad)
            feature_map: [1, 1, 14, 14] - Single feature activation map from CSAE

        Returns:
            saliency: [224, 224] - Saliency map
        """
        # Clear previous gradients
        self.model.zero_grad()
        if image.grad is not None:
            image.grad.zero_()

        # Forward pass through ResNet18
        image.requires_grad_(True)
        _ = self.model(image)

        # Get layer3 activations [1, 256, 14, 14]
        layer3_acts = self.layer3_activations

        # We want to backprop from layer3 activations weighted by feature_map
        # This tells us which layer3 channels (and thus input pixels) contributed to this feature

        # Compute weighted sum: we want gradient of feature_map w.r.t. input
        # Feature map comes from: feature = sum over channels of (layer3_acts * decoder_weights)
        # For simplicity, we'll compute gradient of feature_map's total activation w.r.t. input

        # Sum of feature activations
        target = feature_map.sum()

        # Backward pass
        target.backward()

        # Get gradient w.r.t. input
        saliency = image.grad.data.abs().sum(dim=1).squeeze()  # [224, 224]

        return saliency.cpu()


class MultiChannelSAEVisualizer:
    """
    Visualizer for Multi-Channel ConvSAE learned features.

    For each input image:
    1. Extracts ResNet18 layer3 activations
    2. Passes through CSAE to get sparse features
    3. Identifies top-k activated features
    4. Uses deconvolution to trace back to input pixels
    """

    def __init__(self,
                 csae_model_path: str = 'multichannel_csae_resnet18_model.pkl',
                 device='cuda',
                 cumulative_threshold=0.8):
        """
        Args:
            csae_model_path: Path to trained Multi-Channel ConvSAE
            device: Device to run on
            cumulative_threshold: GradCAM cumulative threshold for channel selection (default: 0.8 = 80%)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.cumulative_threshold = cumulative_threshold

        # Load Multi-Channel ConvSAE
        print(f"Loading Multi-Channel ConvSAE from {csae_model_path}...")
        self.csae_model = joblib.load(csae_model_path).to(self.device)
        self.csae_model.eval()
        print(f"  ✓ Model loaded: {self.csae_model.in_channels}→{self.csae_model.hidden_dim}, top_k={self.csae_model.top_k}")

        # Load ResNet18 backbone
        print("Loading ResNet18 backbone...")
        self.resnet = models.resnet18(pretrained=True).to(self.device)
        self.resnet.eval()

        # Hook for layer3 activations
        self.layer3_activations = None
        self.resnet.layer3.register_forward_hook(self._save_layer3_activation)

        # GradCAM for channel selection
        print("Setting up GradCAM for channel selection...")
        self.gradcam = GradCAM(self.resnet, self.resnet.layer3)
        print(f"  ✓ GradCAM threshold: {cumulative_threshold * 100:.0f}% cumulative score")

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # DeconvNet for visualization
        self.deconv = DeconvNet(self.resnet, device=self.device)

        print("✓ Visualizer ready!\n")

    def _save_layer3_activation(self, module, input, output):
        """Hook to save layer3 activations."""
        self.layer3_activations = output.detach()

    def _select_top_channel_with_gradcam(self, image: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Use GradCAM to select the top channel (highest scoring).

        Args:
            image: [1, 3, 224, 224] - Input image

        Returns:
            top_channel_idx: Index of the top channel
            channel_weights: [256] - GradCAM importance scores for all channels
        """
        # Compute GradCAM channel weights
        weights, _, pred_class = self.gradcam.forward(image, class_idx=None, verbose=False)

        # weights: [256] - GradCAM importance scores (already ReLU'd)

        # Find the top channel
        top_channel_idx = torch.argmax(weights).item()

        return top_channel_idx, weights

    def _normalize_layer3_activations(self, acts: torch.Tensor) -> torch.Tensor:
        """
        Normalize layer3 activations using per-channel 99th percentile.

        Args:
            acts: [1, 256, 14, 14] - Raw layer3 activations

        Returns:
            normalized: [1, 256, 14, 14] - Normalized activations
        """
        normalized = acts.clone()

        # Normalize each channel independently
        for c in range(acts.shape[1]):
            channel_data = acts[0, c, :, :]
            scale_factor = torch.quantile(channel_data.flatten(), 0.99)

            if scale_factor > 1e-8:
                channel_data = torch.clamp(channel_data, min=0.0, max=scale_factor)
                normalized[0, c, :, :] = channel_data / (scale_factor + 1e-8)

        return normalized

    def extract_features(self, image_path: str, top_k: int = 16) -> Dict:
        """
        Extract top-k activated features for an input image using GradCAM channel selection.

        For each image:
        1. Use GradCAM to identify the top activation channel (highest scoring)
        2. Extract that single channel from layer3
        3. Pass through CSAE to get sparse features
        4. Identify top-k activated CSAE features

        Args:
            image_path: Path to input image
            top_k: Number of top features to extract

        Returns:
            results: Dictionary containing:
                - image: Original PIL image
                - image_tensor: Preprocessed image tensor
                - layer3_acts: Full layer3 activations [1, 256, 14, 14]
                - top_channel_idx: Index of the top GradCAM channel
                - top_channel_act: Top channel activation [1, 1, 14, 14]
                - channel_weights: GradCAM weights for all channels [256]
                - sparse_features: CSAE sparse features [1, 4096, 14, 14]
                - top_features: List of (feature_idx, importance, activation_map)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract layer3 activations
        with torch.no_grad():
            _ = self.resnet(image_tensor)
            layer3_acts = self.layer3_activations.clone()  # [1, 256, 14, 14]

        # Use GradCAM to select the top channel
        top_channel_idx, channel_weights = self._select_top_channel_with_gradcam(image_tensor)

        print(f"  GradCAM selected top channel: {top_channel_idx} (score: {channel_weights[top_channel_idx]:.4f})")

        # Extract only the top channel
        top_channel_act = layer3_acts[:, top_channel_idx:top_channel_idx+1, :, :]  # [1, 1, 14, 14]

        # Normalize the top channel activation (same as training)
        top_channel_norm = self._normalize_layer3_activations(top_channel_act)

        # For CSAE: We need to pass a [1, 256, 14, 14] tensor with only the top channel non-zero
        # Create a zero tensor and insert the top channel at its position
        layer3_masked = torch.zeros_like(layer3_acts)
        layer3_masked[:, top_channel_idx:top_channel_idx+1, :, :] = top_channel_norm

        # Pass through CSAE encoder (with Top-K)
        with torch.no_grad():
            _, sparse_features = self.csae_model(layer3_masked, use_topk=True)

        # Compute feature importance (sum of activations per feature)
        # sparse_features: [1, 4096, 14, 14]
        feature_importance = sparse_features.sum(dim=(2, 3)).squeeze()  # [4096]

        # Get top-k features
        top_k_values, top_k_indices = torch.topk(feature_importance, k=min(top_k, len(feature_importance)))

        # Extract activation maps for top features
        top_features = []
        for idx, importance in zip(top_k_indices, top_k_values):
            activation_map = sparse_features[0, idx, :, :].cpu()  # [14, 14]
            top_features.append((idx.item(), importance.item(), activation_map))

        results = {
            'image': image,
            'image_tensor': image_tensor,
            'layer3_acts': layer3_acts.cpu(),
            'top_channel_idx': top_channel_idx,
            'top_channel_act': top_channel_act.cpu(),
            'channel_weights': channel_weights.cpu(),
            'sparse_features': sparse_features.cpu(),
            'top_features': top_features,
            'feature_importance': feature_importance.cpu()
        }

        return results

    def compute_feature_saliency(self, image_tensor: torch.Tensor,
                                 feature_idx: int,
                                 activation_map: torch.Tensor) -> torch.Tensor:
        """
        Compute saliency map for a specific feature using gradient-based method.

        Note: Uses continuous activations (without Top-K) for gradient computation
        since Top-K is non-differentiable. The saliency still shows which input
        regions contribute to this feature's activation.

        Args:
            image_tensor: [1, 3, 224, 224] - Preprocessed input image
            feature_idx: Index of the feature to visualize
            activation_map: [14, 14] - Feature activation map (unused, kept for API compatibility)

        Returns:
            saliency: [224, 224] - Saliency map showing input pixel importance
        """
        # Enable gradients
        image_tensor = image_tensor.clone().requires_grad_(True)

        # Forward pass through ResNet18 (without detaching)
        # We need to manually extract layer3 output without hooks to preserve gradients
        x = image_tensor

        # ResNet18 forward up to layer3
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        layer3_acts = self.resnet.layer3(x)  # [1, 256, 14, 14] - with gradients!

        # Normalize activations (preserve gradients)
        layer3_acts_norm = self._normalize_layer3_activations_with_grad(layer3_acts)

        # Forward through CSAE encoder (with Top-K)
        # Note: Top-K operation is not differentiable, so we'll use the continuous version
        _, sparse_features = self.csae_model(layer3_acts_norm, use_topk=False)  # Don't use top-k for gradients

        # Get the target feature
        target_feature = sparse_features[0, feature_idx, :, :]  # [14, 14]

        # Compute loss: sum of activations for this feature
        loss = target_feature.sum()

        # Backward pass
        self.resnet.zero_grad()
        self.csae_model.zero_grad()
        if image_tensor.grad is not None:
            image_tensor.grad.zero_()

        loss.backward()

        # Get gradient w.r.t. input image
        if image_tensor.grad is None:
            # If gradient is still None, return zeros
            print(f"Warning: No gradient computed for feature {feature_idx}")
            return torch.zeros(224, 224)

        saliency = image_tensor.grad.data.abs().sum(dim=1).squeeze()  # [224, 224]

        return saliency.cpu()

    def _normalize_layer3_activations_with_grad(self, acts: torch.Tensor) -> torch.Tensor:
        """
        Normalize layer3 activations preserving gradients.

        Args:
            acts: [1, 256, 14, 14] - Raw layer3 activations

        Returns:
            normalized: [1, 256, 14, 14] - Normalized activations (with gradients)
        """
        normalized = acts.clone()

        # Normalize each channel independently
        for c in range(acts.shape[1]):
            channel_data = acts[0, c, :, :]

            # Use max instead of quantile for differentiability
            scale_factor = channel_data.max()  # Keep as tensor to preserve gradients

            if scale_factor.item() > 1e-8:
                # Clamp operation (differentiable) - use torch.clamp with proper types
                channel_data = torch.clamp(channel_data, min=0.0)  # Ensure non-negative
                channel_data = torch.minimum(channel_data, scale_factor)  # Clip to max
                normalized[0, c, :, :] = channel_data / (scale_factor + 1e-8)

        return normalized

    def visualize_top_features(self, image_path: str, top_k: int = 16,
                              save_path: str = None):
        """
        Visualize top-k activated features and their input saliency maps.

        Args:
            image_path: Path to input image
            top_k: Number of top features to visualize
            save_path: Path to save visualization (optional)
        """
        print(f"Processing image: {image_path}")
        print(f"Extracting top-{top_k} features...")

        # Extract features
        results = self.extract_features(image_path, top_k=top_k)

        image = results['image']
        image_tensor = results['image_tensor']
        top_features = results['top_features']
        top_channel_idx = results['top_channel_idx']
        top_channel_act = results['top_channel_act']

        print(f"✓ Found {len(top_features)} top features")
        print(f"Computing saliency maps...")

        # Compute saliency maps for each top feature
        saliency_maps = []
        for i, (feat_idx, importance, activation_map) in enumerate(top_features):
            print(f"  Feature {i+1}/{len(top_features)}: #{feat_idx} (importance: {importance:.4f})", end='\r')
            saliency = self.compute_feature_saliency(image_tensor, feat_idx, activation_map)
            saliency_maps.append(saliency)

        print(f"\n✓ Computed {len(saliency_maps)} saliency maps")

        # Visualize
        print("Generating visualization...")
        self._plot_results(image, top_features, saliency_maps, top_channel_idx, top_channel_act, save_path)

        print(f"✓ Visualization complete!")
        if save_path:
            print(f"  Saved to: {save_path}")

    def _plot_results(self, image: Image.Image,
                     top_features: List[Tuple],
                     saliency_maps: List[torch.Tensor],
                     top_channel_idx: int,
                     top_channel_act: torch.Tensor,
                     save_path: str = None):
        """
        Plot visualization of top features and their saliency maps.

        Layout:
        - Row 1: Original image + top GradCAM channel + top feature importance
        - Rows 2+: Grid of top features showing:
            - Feature activation map (14×14) in hidden space
            - Saliency map overlay on input image
        """
        n_features = len(top_features)

        # Create figure with enough space for all features
        n_cols = 9  # 3 overview + 3 pairs of (activation_map, saliency_overlay)
        n_rows = 1 + (n_features + 2) // 3  # Header row + feature rows (3 features per row)

        fig = plt.figure(figsize=(27, 3.5 * n_rows))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.35, wspace=0.25)

        # Row 0: Overview
        # Column 0-1: Input image
        ax_img = fig.add_subplot(gs[0, 0:2])
        ax_img.imshow(image)
        ax_img.set_title("Input Image", fontsize=12, fontweight='bold')
        ax_img.axis('off')

        # Column 2-3: Top GradCAM channel
        ax_channel = fig.add_subplot(gs[0, 2:4])
        channel_map = top_channel_act.squeeze().numpy()  # [14, 14]
        im_ch = ax_channel.imshow(channel_map, cmap='hot', interpolation='bilinear')
        ax_channel.set_title(f"Top GradCAM Channel #{top_channel_idx}\n(14×14 activation)",
                            fontsize=12, fontweight='bold')
        ax_channel.axis('off')
        plt.colorbar(im_ch, ax=ax_channel, fraction=0.046, pad=0.04)

        # Column 4+: Top feature importance distribution
        ax_bar = fig.add_subplot(gs[0, 4:])
        importances = [imp for _, imp, _ in top_features]
        feature_indices = [f"F{idx}" for idx, _, _ in top_features]
        ax_bar.bar(range(len(importances)), importances, color='steelblue', alpha=0.8, edgecolor='navy')
        ax_bar.set_xlabel('Feature Index', fontsize=10)
        ax_bar.set_ylabel('Importance (sum of activations)', fontsize=10)
        ax_bar.set_title(f'Top-{n_features} CSAE Feature Importance\n(from channel #{top_channel_idx})',
                        fontsize=12, fontweight='bold')
        ax_bar.set_xticks(range(len(importances)))
        ax_bar.set_xticklabels(feature_indices, rotation=45, ha='right', fontsize=8)
        ax_bar.grid(True, alpha=0.3, axis='y')

        # Rows 1+: Individual features (3 columns per feature: activation map + saliency overlay + masked)
        for i, ((feat_idx, importance, activation_map), saliency) in enumerate(zip(top_features, saliency_maps)):
            row = 1 + i // 3  # 3 features per row
            col_offset = (i % 3) * 3  # Each feature takes 3 columns

            # Column 1: Feature activation map in hidden space (14×14)
            ax_act = fig.add_subplot(gs[row, col_offset])
            im_act = ax_act.imshow(activation_map.numpy(), cmap='hot', interpolation='bilinear')
            ax_act.set_title(f"F{feat_idx} Activation\n(14×14 hidden space)", fontsize=8, fontweight='bold')
            ax_act.axis('off')
            cbar_act = plt.colorbar(im_act, ax=ax_act, fraction=0.046, pad=0.04)
            cbar_act.ax.tick_params(labelsize=6)

            # Column 2: Saliency map overlaid on original image
            ax_sal = fig.add_subplot(gs[row, col_offset + 1])
            ax_sal.imshow(image, alpha=0.5)
            saliency_norm = saliency / (saliency.max() + 1e-8)
            im_sal = ax_sal.imshow(saliency_norm, cmap='jet', alpha=0.5)
            ax_sal.set_title(f"F{feat_idx} Saliency\nImportance: {importance:.2f}", fontsize=8, fontweight='bold')
            ax_sal.axis('off')
            cbar_sal = plt.colorbar(im_sal, ax=ax_sal, fraction=0.046, pad=0.04)
            cbar_sal.ax.tick_params(labelsize=6)

            # Column 3: Masked image (show only salient regions)
            ax_masked = fig.add_subplot(gs[row, col_offset + 2])
            mask = saliency_norm > 0.5  # Threshold at 50%
            image_resized = Image.fromarray(np.array(image)).resize((224, 224), Image.BILINEAR)
            masked_img = np.array(image_resized).copy()
            if len(masked_img.shape) == 3:
                mask_3d = np.stack([mask, mask, mask], axis=-1)
                masked_img[~mask_3d] = (masked_img[~mask_3d] * 0.3).astype(np.uint8)
            else:
                masked_img[~mask] = (masked_img[~mask] * 0.3).astype(np.uint8)
            ax_masked.imshow(masked_img)
            ax_masked.set_title(f"F{feat_idx} Salient Regions", fontsize=8, fontweight='bold')
            ax_masked.axis('off')

        # Hide unused subplots
        total_feature_cols = ((n_features + 2) // 3) * 3 * 3  # Round up to 3, then * 3 cols per feature
        for i in range(n_features * 3, total_feature_cols):
            row = 1 + i // n_cols
            col = i % n_cols
            if row < n_rows:
                ax = fig.add_subplot(gs[row, col])
                ax.axis('off')

        plt.suptitle(f'Multi-Channel ConvSAE: Top-{n_features} Activated Features (GradCAM Channel #{top_channel_idx})\n' +
                    f'Analysis based on highest-scoring GradCAM channel | Each row: Activation → Saliency → Masked Regions',
                    fontsize=14, fontweight='bold', y=0.998)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_feature_grid(self, image_path: str, top_k: int = 16,
                              save_path: str = None):
        """
        Create a compact grid visualization showing activation maps and saliency.

        Each row shows one feature with 4 columns:
        - Column 1: Feature activation map in hidden space (14×14)
        - Column 2: Saliency map (224×224)
        - Column 3: Saliency overlay on input image
        - Column 4: Masked image (only salient regions)

        Args:
            image_path: Path to input image
            top_k: Number of top features to visualize
            save_path: Path to save visualization
        """
        print(f"Processing image: {image_path}")

        # Extract features
        results = self.extract_features(image_path, top_k=top_k)
        image = results['image']
        image_tensor = results['image_tensor']
        top_features = results['top_features']
        top_channel_idx = results['top_channel_idx']
        top_channel_act = results['top_channel_act']

        # Compute saliency maps
        print(f"Computing saliency maps for {len(top_features)} features...")
        saliency_maps = []
        for i, (feat_idx, importance, activation_map) in enumerate(top_features):
            print(f"  {i+1}/{len(top_features)}", end='\r')
            saliency = self.compute_feature_saliency(image_tensor, feat_idx, activation_map)
            saliency_maps.append(saliency)
        print()

        # Create compact grid: each row shows [activation_map | saliency | overlay | masked]
        # First row: Input image and top GradCAM channel
        n_features = len(top_features)
        fig, axes = plt.subplots(n_features + 1, 4, figsize=(18, 3 * (n_features + 1)))

        # Row 0: Overview
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Input Image', fontsize=10, fontweight='bold')
        axes[0, 0].axis('off')

        channel_map = top_channel_act.squeeze().numpy()
        im_ch = axes[0, 1].imshow(channel_map, cmap='hot', interpolation='bilinear')
        axes[0, 1].set_title(f'Top GradCAM Channel #{top_channel_idx}\n(14×14 activation)',
                            fontsize=10, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im_ch, ax=axes[0, 1], fraction=0.046, pad=0.04)

        axes[0, 2].axis('off')
        axes[0, 3].text(0.5, 0.5, f'Analyzing {n_features} CSAE features\nfrom channel #{top_channel_idx}',
                       ha='center', va='center', fontsize=11, fontweight='bold',
                       transform=axes[0, 3].transAxes)
        axes[0, 3].axis('off')

        # Rows 1+: Features
        for i, ((feat_idx, importance, activation_map), saliency) in enumerate(zip(top_features, saliency_maps)):
            row_idx = i + 1  # Offset by 1 for header row

            # Column 0: Feature activation map in hidden space (14×14)
            im0 = axes[row_idx, 0].imshow(activation_map.numpy(), cmap='hot', interpolation='bilinear')
            axes[row_idx, 0].set_title(f"Feature {feat_idx}\nActivation (14×14)\nImportance: {importance:.2f}",
                                fontsize=9, fontweight='bold')
            axes[row_idx, 0].axis('off')
            plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

            # Column 1: Saliency map (224×224)
            saliency_norm = saliency / (saliency.max() + 1e-8)
            im1 = axes[row_idx, 1].imshow(saliency_norm, cmap='jet')
            axes[row_idx, 1].set_title(f"Saliency Map\n(224×224)", fontsize=9, fontweight='bold')
            axes[row_idx, 1].axis('off')
            plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

            # Column 2: Overlay on image
            axes[row_idx, 2].imshow(image, alpha=0.6)
            axes[row_idx, 2].imshow(saliency_norm, cmap='jet', alpha=0.4)
            axes[row_idx, 2].set_title(f"Saliency Overlay\non Input", fontsize=9, fontweight='bold')
            axes[row_idx, 2].axis('off')

            # Column 3: Masked image (show only salient regions)
            mask = saliency_norm > 0.5  # Threshold at 50%, shape (224, 224)

            # Resize image to match mask dimensions (224×224)
            image_resized = Image.fromarray(np.array(image)).resize((224, 224), Image.BILINEAR)
            masked_img = np.array(image_resized).copy()

            # Apply mask (ensure mask has same shape as image)
            if len(masked_img.shape) == 3:  # RGB image
                mask_3d = np.stack([mask, mask, mask], axis=-1)
                masked_img[~mask_3d] = (masked_img[~mask_3d] * 0.3).astype(np.uint8)
            else:  # Grayscale
                masked_img[~mask] = (masked_img[~mask] * 0.3).astype(np.uint8)

            axes[row_idx, 3].imshow(masked_img)
            axes[row_idx, 3].set_title(f"Salient Regions\n(threshold=0.5)", fontsize=9, fontweight='bold')
            axes[row_idx, 3].axis('off')

        plt.suptitle(f'Multi-Channel ConvSAE Feature Analysis (Grid View) - GradCAM Channel #{top_channel_idx}\n' +
                    f'Image: {Path(image_path).name} | Showing Top-{n_features} Features',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved to: {save_path}")
            plt.close()
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Multi-Channel ConvSAE learned features'
    )
    parser.add_argument('--csae_model', type=str,
                       default='multichannel_csae_resnet18_model.pkl',
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
    parser.add_argument('--output_dir', type=str, default='multichannel_sae_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--grid_view', action='store_true',
                       help='Use compact grid view instead of default view')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create visualizer
    print("="*80)
    print("Multi-Channel ConvSAE Feature Visualization")
    print("="*80 + "\n")

    visualizer = MultiChannelSAEVisualizer(
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
        if args.grid_view:
            save_path = output_dir / f"{img_name}_grid.png"
            visualizer.visualize_feature_grid(
                str(img_path),
                top_k=args.top_k_features,
                save_path=str(save_path)
            )
        else:
            save_path = output_dir / f"{img_name}_features.png"
            visualizer.visualize_top_features(
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
