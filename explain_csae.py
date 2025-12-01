"""
Explainability Module using Convolutional Sparse Autoencoder (ConvSAE)

This module provides tools to explain model predictions by:
1. Using GradCAM to find top influential activation maps
2. Decomposing activation maps using trained ConvSAE encoder
3. Identifying top-k activated CSAE features
4. Visualizing the learned features and their spatial activations

Usage:
    python explain_csae.py --image_path data/imagenette/tench/n01440764_1.JPEG
    python explain_csae.py --class_name tench --num_images 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from typing import Dict, List, Tuple, Optional
import joblib
import argparse
from pathlib import Path

from src.gradcam import GradCAM
from src.model import FineTunedModel
from src.convsae import ConvSAE
from torchvision import transforms


class CSAEExplainer:
    """
    Explanation pipeline using trained ConvSAE.

    Args:
        model: Trained classification model (e.g., FineTunedModel)
        csae_model: Trained ConvSAE model
        device: Computing device (cuda/cpu)
        target_layer: Layer to extract activations from (default: layer 5)
    """

    def __init__(
        self,
        model: nn.Module,
        csae_model: ConvSAE,
        device: torch.device,
        target_layer: Optional[nn.Module] = None,
        training_info: Optional[Dict] = None
    ):
        self.model = model.to(device)
        self.model.eval()
        self.csae_model = csae_model.to(device)
        self.csae_model.eval()
        self.device = device
        self.training_info = training_info

        # Default to layer 5 of feature extractor
        if target_layer is None:
            self.target_layer = model.feature_extractor[5]
        else:
            self.target_layer = target_layer

        # Initialize GradCAM
        self.gradcam = GradCAM(model, self.target_layer)

        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # For denormalization
        self.denormalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    def load_and_preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """Load and preprocess an image."""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        return img_tensor, img

    def get_top_k_cumulative(
        self,
        weights: torch.Tensor,
        cumulative_threshold: float = 0.8
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Get top-k channel indices based on cumulative score threshold.

        Args:
            weights: Channel weights from GradCAM [C]
            cumulative_threshold: Cumulative weight threshold (default 0.8)

        Returns:
            selected_indices: List of channel indices
            cumulative_weights: Cumulative weights for selected channels
        """
        # Sort by absolute weight
        abs_weights = weights.abs()
        sorted_indices = torch.argsort(abs_weights, descending=True)
        sorted_weights = abs_weights[sorted_indices]

        # Calculate cumulative sum
        total_weight = sorted_weights.sum()
        cumulative_sum = torch.cumsum(sorted_weights, dim=0)
        cumulative_ratio = cumulative_sum / total_weight

        # Find channels that contribute to cumulative_threshold
        n_selected = (cumulative_ratio <= cumulative_threshold).sum().item() + 1
        n_selected = min(n_selected, len(sorted_indices))

        selected_indices = sorted_indices[:n_selected].tolist()
        return selected_indices, cumulative_ratio[:n_selected]

    def apply_robust_normalization(self, activation_map: torch.Tensor) -> torch.Tensor:
        """Apply robust normalization to activation map (same as training)."""
        # Robust scaling using 99th percentile
        scale_factor = torch.quantile(activation_map.flatten(), 0.99)
        normalized = torch.clamp(activation_map, min=0.0, max=scale_factor)
        normalized = normalized / (scale_factor + 1e-8)
        normalized = normalized * 10
        return normalized

    def explain_prediction(
        self,
        image_path: str,
        class_idx: Optional[int] = None,
        cumulative_threshold: float = 0.8,
        top_k_features: int = 10
    ) -> Dict:
        """
        Explain prediction for a single image using ConvSAE.

        Args:
            image_path: Path to input image
            class_idx: Target class index (None = predicted class)
            cumulative_threshold: Threshold for selecting top activation maps
            top_k_features: Number of top CSAE features to show

        Returns:
            Dictionary containing explanation results
        """
        # Load image
        img_tensor, original_img = self.load_and_preprocess_image(image_path)

        # Get prediction
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = F.softmax(logits, dim=1)
            predicted_class = logits.argmax(dim=1).item()
            predicted_prob = probs[0, predicted_class].item()

        if class_idx is None:
            class_idx = predicted_class

        print(f"\nPredicted class: {predicted_class} (prob: {predicted_prob:.3f})")
        print(f"Explaining class: {class_idx}")

        # Get GradCAM weights and activation map
        channel_weights, cam_map, _ = self.gradcam.forward(img_tensor, class_idx)

        # Get activation map from target layer (already stored in gradcam after forward pass)
        activation_maps = self.gradcam.activations  # [1, C, H, W]

        # Select top channels using cumulative threshold
        selected_channels, _ = self.get_top_k_cumulative(
            channel_weights, cumulative_threshold
        )

        print(f"Selected {len(selected_channels)} channels (cumulative threshold: {cumulative_threshold})")

        # Process each selected channel through CSAE
        csae_results = []
        for ch_idx in selected_channels:
            # Extract single channel activation map
            act_map = activation_maps[0:1, ch_idx:ch_idx+1, :, :]  # [1, 1, H, W]

            # Apply robust normalization (same as training)
            act_map_norm = self.apply_robust_normalization(act_map)

            # Pass through CSAE encoder
            with torch.no_grad():
                reconstruction, sparse_features = self.csae_model(act_map_norm)

            # Get top-k activated features for this channel
            # sparse_features: [1, hidden_dim, H, W]
            feature_importance = sparse_features.sum(dim=(2, 3)).squeeze()  # [hidden_dim]
            top_features = torch.topk(feature_importance, min(top_k_features, feature_importance.shape[0]))

            csae_results.append({
                'channel_idx': ch_idx,
                'channel_weight': channel_weights[ch_idx].item(),
                'activation_map': act_map.cpu(),
                'activation_map_norm': act_map_norm.cpu(),
                'reconstruction': reconstruction.cpu(),
                'sparse_features': sparse_features.cpu(),
                'top_feature_indices': top_features.indices.cpu().numpy(),
                'top_feature_values': top_features.values.cpu().numpy()
            })

        return {
            'image_path': image_path,
            'original_image': original_img,
            'img_tensor': img_tensor.cpu(),
            'predicted_class': predicted_class,
            'predicted_prob': predicted_prob,
            'explained_class': class_idx,
            'cam_map': cam_map.cpu().numpy(),
            'channel_weights': channel_weights.cpu(),
            'selected_channels': selected_channels,
            'csae_results': csae_results
        }

    def visualize_explanation(
        self,
        results: Dict,
        save_path: str = 'csae_explanation.png',
        max_channels: int = 5
    ):
        """
        Visualize the CSAE explanation.

        Args:
            results: Dictionary from explain_prediction()
            save_path: Path to save visualization
            max_channels: Maximum number of channels to show
        """
        csae_results = results['csae_results'][:max_channels]
        n_channels = len(csae_results)

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 4 * n_channels + 3))
        gs = GridSpec(n_channels + 1, 5, figure=fig, hspace=0.4, wspace=0.3)

        # Top row: Original image and GradCAM
        ax_img = fig.add_subplot(gs[0, 0:2])
        ax_cam = fig.add_subplot(gs[0, 2:4])

        # Show original image
        img = results['original_image']
        ax_img.imshow(img)
        ax_img.set_title(
            f"Input Image\nPredicted: {results['predicted_class']} "
            f"(prob: {results['predicted_prob']:.3f})",
            fontsize=12, fontweight='bold'
        )
        ax_img.axis('off')

        # Show GradCAM
        ax_cam.imshow(img)
        ax_cam.imshow(results['cam_map'], cmap='jet', alpha=0.4)
        ax_cam.set_title(
            f"GradCAM (Layer 5)\nExplaining class: {results['explained_class']}",
            fontsize=12, fontweight='bold'
        )
        ax_cam.axis('off')

        # Show number of channels
        ax_info = fig.add_subplot(gs[0, 4])
        ax_info.axis('off')
        ax_info.text(0.1, 0.5,
                    f"Selected {len(results['selected_channels'])} channels\n"
                    f"Showing top {n_channels}",
                    fontsize=11, verticalalignment='center')

        # For each selected channel
        for i, csae_result in enumerate(csae_results):
            row = i + 1
            ch_idx = csae_result['channel_idx']
            ch_weight = csae_result['channel_weight']

            # Column 0: Original activation map
            ax_act = fig.add_subplot(gs[row, 0])
            act_map = csae_result['activation_map'].squeeze().numpy()
            im_act = ax_act.imshow(act_map, cmap='hot')
            ax_act.set_title(f"Channel {ch_idx}\nWeight: {ch_weight:.3f}", fontsize=10)
            ax_act.axis('off')
            plt.colorbar(im_act, ax=ax_act, fraction=0.046)

            # Column 1: Normalized activation map
            ax_norm = fig.add_subplot(gs[row, 1])
            act_map_norm = csae_result['activation_map_norm'].squeeze().numpy()
            im_norm = ax_norm.imshow(act_map_norm, cmap='hot')
            ax_norm.set_title("Normalized", fontsize=10)
            ax_norm.axis('off')
            plt.colorbar(im_norm, ax=ax_norm, fraction=0.046)

            # Column 2: CSAE Reconstruction
            ax_recon = fig.add_subplot(gs[row, 2])
            recon = csae_result['reconstruction'].squeeze().numpy()
            im_recon = ax_recon.imshow(recon, cmap='hot')
            mse = ((act_map_norm - recon) ** 2).mean()
            ax_recon.set_title(f"CSAE Recon\nMSE: {mse:.4f}", fontsize=10)
            ax_recon.axis('off')
            plt.colorbar(im_recon, ax=ax_recon, fraction=0.046)

            # Column 3: Top CSAE features (decoder weights)
            ax_features = fig.add_subplot(gs[row, 3])
            top_indices = csae_result['top_feature_indices'][:5]
            top_values = csae_result['top_feature_values'][:5]

            # Get decoder weights for top features
            decoder_weights = self.csae_model.decoder.weight.detach().cpu()  # [1, hidden_dim, 1, 1]
            top_weights = decoder_weights[0, top_indices, 0, 0].numpy()

            # Bar plot
            ax_features.barh(range(len(top_indices)), top_values, color='steelblue')
            ax_features.set_yticks(range(len(top_indices)))
            ax_features.set_yticklabels([f"F{idx}\n(w={w:.2f})" for idx, w in zip(top_indices, top_weights)])
            ax_features.set_xlabel("Activation Sum", fontsize=9)
            ax_features.set_title("Top 5 CSAE Features", fontsize=10)
            ax_features.invert_yaxis()

            # Column 4: Spatial activation pattern of top feature
            ax_spatial = fig.add_subplot(gs[row, 4])
            sparse_features = csae_result['sparse_features'].squeeze()  # [hidden_dim, H, W]
            top_feature_idx = top_indices[0]
            spatial_pattern = sparse_features[top_feature_idx].numpy()
            im_spatial = ax_spatial.imshow(spatial_pattern, cmap='viridis')
            ax_spatial.set_title(f"Feature {top_feature_idx}\nSpatial Pattern", fontsize=10)
            ax_spatial.axis('off')
            plt.colorbar(im_spatial, ax=ax_spatial, fraction=0.046)

        plt.suptitle(
            f"ConvSAE Explanation: {Path(results['image_path']).name}",
            fontsize=16, fontweight='bold', y=0.995
        )

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nExplanation saved to {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Explain model predictions using ConvSAE')
    parser.add_argument('--image_path', type=str, help='Path to input image')
    parser.add_argument('--class_name', type=str, help='Class name to analyze multiple images')
    parser.add_argument('--num_images', type=int, default=1, help='Number of images to analyze')
    parser.add_argument('--model_path', type=str, default='weights/finetune_weights.pth',
                       help='Path to model weights')
    parser.add_argument('--csae_path', type=str, default='csae_model.pkl',
                       help='Path to trained CSAE model')
    parser.add_argument('--cumulative_threshold', type=float, default=0.8,
                       help='Cumulative threshold for channel selection')
    parser.add_argument('--top_k_features', type=int, default=10,
                       help='Number of top CSAE features to analyze')
    parser.add_argument('--max_channels', type=int, default=5,
                       help='Maximum channels to visualize')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = FineTunedModel(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Load CSAE
    print("Loading ConvSAE...")
    csae_model = joblib.load(args.csae_path)
    csae_model = csae_model.to(device)
    csae_model.eval()

    # Load training info if available
    training_info = None
    training_info_path = 'csae_training_info.pkl'
    if Path(training_info_path).exists():
        training_info = joblib.load(training_info_path)
        print(f"CSAE training config: {training_info['config']}")

    # Create explainer
    explainer = CSAEExplainer(model, csae_model, device, training_info=training_info)

    # Get image paths
    if args.image_path:
        image_paths = [args.image_path]
    elif args.class_name:
        class_dir = Path(f'data/imagenette/{args.class_name}')
        if not class_dir.exists():
            raise ValueError(f"Class directory not found: {class_dir}")
        image_paths = sorted(list(class_dir.glob('*.JPEG')))[:args.num_images]
    else:
        raise ValueError("Must provide either --image_path or --class_name")

    # Process each image
    for img_path in image_paths:
        print(f"\n{'='*70}")
        print(f"Processing: {img_path}")
        print('='*70)

        results = explainer.explain_prediction(
            str(img_path),
            cumulative_threshold=args.cumulative_threshold,
            top_k_features=args.top_k_features
        )

        # Create output filename
        output_name = f"csae_explanation_{Path(img_path).stem}.png"
        explainer.visualize_explanation(
            results,
            save_path=output_name,
            max_channels=args.max_channels
        )


if __name__ == "__main__":
    main()
