"""
Top Activation Map Visualization with Gradient-Based Saliency

This script:
1. Extracts top activation maps from layer 5 using GradCAM scores (80% cumulative)
2. Uses gradient-based saliency to trace which input image regions activate each map
3. Visualizes the activation maps and their corresponding input regions

The saliency is computed as ∂(activation_sum)/∂(input), showing which pixels
contribute most to each channel's activation.

Usage:
    python view_top_activation.py --image_path data/imagenette/tench/n01440764_1.JPEG
    python view_top_activation.py --class_name tench --num_images 5
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from torchvision import datasets, transforms
from typing import Dict, List, Tuple, Optional
import argparse
import os

from src.gradcam import GradCAM
from src.model import FineTunedModel


class DeconvNet:
    """
    Deconvolutional Network for visualizing what activates specific feature maps.

    Uses gradient-based saliency to compute which input regions contribute most
    to each activation map. Computes ∂(activation_sum)/∂(input) for attribution.

    Note: Uses standard gradients instead of guided backprop to avoid conflicts
    with the model's inplace ReLU operations.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Storage for activations and gradients
        self.activations = {}
        self.gradients = {}

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks for storing activations (forward only to avoid inplace conflicts)"""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach().clone()
            return hook

        # Register forward hooks only (backward hooks conflict with inplace ReLU)
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(forward_hook(f'conv_{name}'))

    def compute_activation_deconv(
        self,
        image: torch.Tensor,
        target_layer: nn.Module,
        channel_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute deconvolution for a specific channel in target layer.

        Args:
            image: Input image [1, 3, H, W]
            target_layer: Layer containing the activation map
            channel_idx: Which channel to visualize

        Returns:
            activation_map: The activation map [H_act, W_act]
            deconv_map: Deconvolution saliency map [H_img, W_img]
        """
        image = image.to(self.device)
        image_input = image.clone().requires_grad_(True)

        # Storage for target activation
        target_activation = None

        def hook_target(module, input, output):
            nonlocal target_activation
            target_activation = output

        # Register hook on target layer
        handle = target_layer.register_forward_hook(hook_target)

        try:
            # Forward pass
            with torch.enable_grad():
                output = self.model(image_input)

                if target_activation is None:
                    raise ValueError("Target activation not captured")

                # Get the specific channel activation
                channel_activation = target_activation[0, channel_idx]  # [H, W]

                # Backward pass: maximize the mean activation of this channel
                activation_score = channel_activation.sum()

                self.model.zero_grad()
                activation_score.backward()

                # Get gradient w.r.t. input
                if image_input.grad is not None:
                    gradient = image_input.grad.detach()

                    # Compute saliency as absolute gradient magnitude
                    saliency = gradient.abs().sum(dim=1).squeeze()  # [H_img, W_img]

                    # Normalize
                    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
                else:
                    saliency = torch.zeros(image.shape[2:], device=self.device)

                # Get activation map
                act_map = channel_activation.detach().cpu()

        finally:
            handle.remove()

        return act_map, saliency.cpu()


class TopActivationVisualizer:
    """
    Visualizes top activation maps and their input regions using gradient-based saliency.

    For each top activation channel, computes which input pixels contribute most
    to that channel's activation via backpropagation.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        target_layer: Optional[nn.Module] = None
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Default to layer 5
        if target_layer is None:
            self.target_layer = model.feature_extractor[5]
        else:
            self.target_layer = target_layer

        # Initialize GradCAM and DeconvNet
        self.gradcam = GradCAM(model, self.target_layer)
        self.deconv = DeconvNet(model, device)

    def analyze_image(
        self,
        image: torch.Tensor,
        class_idx: Optional[int] = None,
        cumulative_threshold: float = 0.8
    ) -> Dict:
        """
        Analyze top activation maps for an image.

        Args:
            image: Input image [1, 3, H, W]
            class_idx: Target class (None = predicted class)
            cumulative_threshold: Cumulative score threshold (default: 0.8)

        Returns:
            results: Dictionary containing:
                - 'input_image': Input image
                - 'predicted_class': Predicted/target class
                - 'gradcam_weights': Channel importance weights
                - 'top_indices': Selected channel indices
                - 'activation_maps': List of activation maps
                - 'deconv_maps': List of deconvolution saliency maps
                - 'gradcam_heatmap': Overall GradCAM heatmap
        """
        image = image.to(self.device)

        # Step 1: Get GradCAM weights
        print("Computing GradCAM weights...")
        weights, cam, predicted_class = self.gradcam.forward(image, class_idx=class_idx)

        # Step 2: Select top channels (80% cumulative)
        sorted_weights, sorted_indices = torch.sort(weights, descending=True)
        cumulative_sum = torch.cumsum(sorted_weights, dim=0)
        total_weight = sorted_weights.sum()
        cumulative_scores = cumulative_sum / total_weight

        cutoff_idx = torch.where(cumulative_scores >= cumulative_threshold)[0]
        if len(cutoff_idx) > 0:
            k = cutoff_idx[0].item() + 1
        else:
            k = len(weights)

        top_indices = sorted_indices[:k].cpu().tolist()

        print(f"Selected {len(top_indices)} / {len(weights)} channels")
        print(f"Cumulative score: {cumulative_scores[k-1]:.2%}")

        # Step 3: Get activations
        activations = self.gradcam.activations.squeeze()  # [C, H, W]

        # Step 4: Compute gradient-based saliency for each top channel
        print(f"\nComputing saliency maps for {len(top_indices)} channels...")
        activation_maps = []
        deconv_maps = []

        for idx, channel_idx in enumerate(top_indices):
            print(f"  [{idx+1}/{len(top_indices)}] Channel {channel_idx}...", end='')

            # Get activation map
            act_map = activations[channel_idx].cpu()

            # Compute gradient-based saliency
            _, deconv_map = self.deconv.compute_activation_deconv(
                image,
                self.target_layer,
                channel_idx
            )

            activation_maps.append(act_map)
            deconv_maps.append(deconv_map)

            print(f" ✓")

        return {
            'input_image': image.cpu(),
            'predicted_class': predicted_class,
            'gradcam_weights': weights.cpu(),
            'top_indices': top_indices,
            'cumulative_scores': cumulative_scores.cpu(),
            'activation_maps': activation_maps,
            'deconv_maps': deconv_maps,
            'gradcam_heatmap': cam.cpu()
        }

    def visualize_results(
        self,
        results: Dict,
        save_path: Optional[str] = None,
        max_channels: int = 8
    ):
        """
        Visualize top activation maps and their gradient-based saliency results.

        Creates a grid showing:
        - Row 0: Input image, GradCAM, overlay
        - Rows 1+: For each channel: activation map, gradient saliency, overlay

        Args:
            results: Results from analyze_image()
            save_path: Path to save figure
            max_channels: Maximum number of channels to show
        """
        n_channels = min(max_channels, len(results['activation_maps']))

        # Create figure
        fig = plt.figure(figsize=(16, 3 * (n_channels + 1)))
        gs = GridSpec(n_channels + 1, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Denormalize input image
        img_tensor = results['input_image'].squeeze(0).numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_denorm = img_tensor * std + mean
        img_denorm = np.clip(img_denorm, 0, 1)
        img_denorm = np.transpose(img_denorm, (1, 2, 0))

        # ===== Row 0: Overview =====
        # Input image
        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(img_denorm)
        ax_img.set_title(f'Input Image\nClass: {results["predicted_class"]}',
                        fontsize=12, fontweight='bold')
        ax_img.axis('off')

        # GradCAM heatmap
        ax_gradcam = fig.add_subplot(gs[0, 1])
        gradcam_heatmap = results['gradcam_heatmap'].numpy()
        ax_gradcam.imshow(gradcam_heatmap, cmap='jet')
        ax_gradcam.set_title('GradCAM Heatmap\n(Layer 5)', fontsize=12, fontweight='bold')
        ax_gradcam.axis('off')

        # Overlay
        ax_overlay = fig.add_subplot(gs[0, 2:4])
        from PIL import Image as PILImage
        H_img, W_img = img_denorm.shape[:2]
        H_heat, W_heat = gradcam_heatmap.shape

        heat_pil = PILImage.fromarray((gradcam_heatmap * 255).astype(np.uint8))
        heat_resized = heat_pil.resize((W_img, H_img), PILImage.BILINEAR)
        heat_resized = np.array(heat_resized) / 255.0

        ax_overlay.imshow(img_denorm)
        ax_overlay.imshow(heat_resized, cmap='jet', alpha=0.4)
        ax_overlay.set_title(f'GradCAM Overlay\n({len(results["top_indices"])} channels, 80% cumulative)',
                           fontsize=12, fontweight='bold')
        ax_overlay.axis('off')

        # ===== Rows 1+: Per-channel details =====
        for row_idx in range(n_channels):
            row = row_idx + 1
            channel_idx = results['top_indices'][row_idx]
            channel_weight = results['gradcam_weights'][channel_idx].item()
            act_map = results['activation_maps'][row_idx].numpy()
            deconv_map = results['deconv_maps'][row_idx].numpy()

            # Column 0: Activation map
            ax_act = fig.add_subplot(gs[row, 0])
            im = ax_act.imshow(act_map, cmap='viridis')
            ax_act.set_title(f'Channel {channel_idx}\nActivation Map\nWeight: {channel_weight:.4f}',
                           fontsize=10, fontweight='bold')
            ax_act.axis('off')
            plt.colorbar(im, ax=ax_act, fraction=0.046)

            # Column 1: Gradient saliency
            ax_deconv = fig.add_subplot(gs[row, 1])
            im2 = ax_deconv.imshow(deconv_map, cmap='hot')
            ax_deconv.set_title('Gradient Saliency\n(Input Attribution)', fontsize=10, fontweight='bold')
            ax_deconv.axis('off')
            plt.colorbar(im2, ax=ax_deconv, fraction=0.046)

            # Column 2: Overlay on input
            ax_overlay_ch = fig.add_subplot(gs[row, 2])
            ax_overlay_ch.imshow(img_denorm)
            ax_overlay_ch.imshow(deconv_map, cmap='hot', alpha=0.5)
            ax_overlay_ch.set_title('Saliency Overlay\n(Activated Regions)', fontsize=10, fontweight='bold')
            ax_overlay_ch.axis('off')

            # Column 3: Masked input
            ax_masked = fig.add_subplot(gs[row, 3])

            # Threshold at 50% of max
            mask = deconv_map > 0.5
            masked_img = img_denorm.copy()
            for c in range(3):
                masked_img[:, :, c] = masked_img[:, :, c] * mask

            # Blend for context
            blend_alpha = 0.3
            masked_img = blend_alpha * img_denorm + (1 - blend_alpha) * masked_img

            ax_masked.imshow(masked_img)
            ax_masked.set_title('Highlighted Regions\n(Thresholded)', fontsize=10, fontweight='bold')
            ax_masked.axis('off')

        plt.suptitle('Top Activation Maps with Gradient-Based Saliency Analysis',
                    fontsize=14, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Saved visualization to: {save_path}")

        plt.tight_layout()
        plt.show()

    def visualize_grid(
        self,
        results: Dict,
        save_path: Optional[str] = None,
        channels_per_row: int = 4
    ):
        """
        Create a compact grid visualization showing all channels.

        Args:
            results: Results from analyze_image()
            save_path: Path to save figure
            channels_per_row: Number of channels per row
        """
        n_channels = len(results['activation_maps'])
        n_rows = int(np.ceil(n_channels / channels_per_row))

        fig = plt.figure(figsize=(4 * channels_per_row, 3 * n_rows))

        # Denormalize input
        img_tensor = results['input_image'].squeeze(0).numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_denorm = img_tensor * std + mean
        img_denorm = np.clip(img_denorm, 0, 1)
        img_denorm = np.transpose(img_denorm, (1, 2, 0))

        for idx, (channel_idx, act_map, deconv_map) in enumerate(zip(
            results['top_indices'],
            results['activation_maps'],
            results['deconv_maps']
        )):
            ax = plt.subplot(n_rows, channels_per_row, idx + 1)

            # Show overlay
            act_map_np = act_map.numpy()
            deconv_map_np = deconv_map.numpy()

            ax.imshow(img_denorm)
            ax.imshow(deconv_map_np, cmap='hot', alpha=0.6)

            weight = results['gradcam_weights'][channel_idx].item()
            ax.set_title(f'Ch {channel_idx} (w={weight:.3f})', fontsize=10, fontweight='bold')
            ax.axis('off')

        plt.suptitle(f'All {n_channels} Top Activation Channels (80% cumulative)',
                    fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Saved grid visualization to: {save_path}")

        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize top activation maps using deconvolution"
    )
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to a specific image')
    parser.add_argument('--class_name', type=str, default='church',
                       help='Class name to sample from (default: church)')
    parser.add_argument('--num_images', type=int, default=1,
                       help='Number of images to process (default: 1)')
    parser.add_argument('--output_dir', type=str, default='top_activations',
                       help='Output directory for visualizations')
    parser.add_argument('--cumulative_threshold', type=float, default=0.8,
                       help='Cumulative threshold for channel selection (default: 0.8)')
    parser.add_argument('--max_channels_show', type=int, default=8,
                       help='Maximum channels to show in detail view (default: 8)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    print("Loading model...")
    model = FineTunedModel(num_classes=10).to(device)
    model.load_state_dict(torch.load('weights/finetune_weights.pth', map_location=device))
    model.eval()
    print("✓ Model loaded")

    # Create visualizer
    print("\nCreating visualizer...")
    visualizer = TopActivationVisualizer(
        model=model,
        device=device,
        target_layer=model.feature_extractor[5]
    )
    print("✓ Visualizer ready")

    # Process images
    if args.image_path:
        # Single image mode
        print(f"\n{'='*70}")
        print("Single Image Mode")
        print(f"{'='*70}\n")

        # Load image
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = Image.open(args.image_path).convert('RGB')
        img_tensor = data_transform(img).unsqueeze(0)

        # Analyze
        results = visualizer.analyze_image(
            img_tensor,
            cumulative_threshold=args.cumulative_threshold
        )

        # Visualize
        base_name = os.path.splitext(os.path.basename(args.image_path))[0]
        save_path = os.path.join(args.output_dir, f'{base_name}_detailed.png')
        visualizer.visualize_results(results, save_path, max_channels=args.max_channels_show)

        save_path_grid = os.path.join(args.output_dir, f'{base_name}_grid.png')
        visualizer.visualize_grid(results, save_path_grid)

    else:
        # Dataset mode
        print(f"\n{'='*70}")
        print(f"Dataset Mode - Class: {args.class_name}")
        print(f"{'='*70}\n")

        # Load dataset
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(root='data/imagenette', transform=data_transform)
        class_names = dataset.classes

        if args.class_name not in class_names:
            print(f"Error: Class '{args.class_name}' not found.")
            print(f"Available classes: {class_names}")
            return

        class_idx = class_names.index(args.class_name)
        class_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]

        # Sample images
        num_samples = min(args.num_images, len(class_indices))
        sampled_indices = np.random.choice(class_indices, num_samples, replace=False)

        print(f"Processing {num_samples} images from class '{args.class_name}'...\n")

        for idx, img_idx in enumerate(sampled_indices):
            print(f"\n{'='*70}")
            print(f"Image {idx+1}/{num_samples} (index: {img_idx})")
            print(f"{'='*70}\n")

            img_tensor, label = dataset[img_idx]
            img_tensor = img_tensor.unsqueeze(0)

            # Analyze
            results = visualizer.analyze_image(
                img_tensor,
                cumulative_threshold=args.cumulative_threshold
            )

            # Visualize
            save_path = os.path.join(args.output_dir,
                                    f'{args.class_name}_img{idx}_detailed.png')
            visualizer.visualize_results(results, save_path, max_channels=args.max_channels_show)

            save_path_grid = os.path.join(args.output_dir,
                                         f'{args.class_name}_img{idx}_grid.png')
            visualizer.visualize_grid(results, save_path_grid)

    print(f"\n{'='*70}")
    print(f"✓ Complete! Check '{args.output_dir}/' for visualizations.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
