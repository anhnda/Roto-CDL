"""
Same-Class Dual ConvSAE Feature Activation Analysis

This script analyzes whether images from the same class share similar activated features
in BOTH the shared pathway (global features) and class-specific pathway (discriminative features).

Key differences from single ConvSAE analysis:
- Analyzes TWO pathways: shared (global) and class-specific (discriminative)
- Compares how shared vs class features differ in their activation patterns
- Validates that class pathway learns discriminative features while shared pathway learns common patterns

Usage:
    # Analyze a single class
    python check_same_class_dual_csae.py --class_name tench --num_images 10

    # Compare multiple classes
    python check_same_class_dual_csae.py --compare_classes tench church parachute --num_images 10

    # Use ResNet18 backbone
    python check_same_class_dual_csae.py --compare_classes tench church parachute --num_images 10 --use_resnet18
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from torchvision import datasets, transforms
import torchvision.models as models
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse
import os
import joblib

from src.model import FineTunedModel
from src.convsae import DualConvSAE
from src.gradcam import GradCAM


# ==========================================
# ResNet18 Support (for run_dual_csae_resnet18.py)
# ==========================================

IMAGENETTE_TO_IMAGENET = {
    'tench': 0,
    'springer': 217,
    'cassette_player': 482,
    'chain_saw': 491,
    'church': 497,
    'french_horn': 566,
    'garbage_truck': 569,
    'gas_pump': 571,
    'golf_ball': 574,
    'parachute': 701,
}

IMAGENETTE_CLASSES = [
    'tench', 'springer', 'cassette_player', 'chain_saw', 'church',
    'french_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute'
]


class ResNet18Wrapper:
    """Wrapper to make ResNet18 compatible with GradCAM."""
    def __init__(self, device='cuda'):
        self.model = models.resnet18(pretrained=True).to(device)
        self.model.eval()
        self.feature_extractor = self.model.layer3  # Use layer3 as target
        self.device = device

    def __call__(self, x):
        return self.model(x)

    def to(self, device):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self


class SameClassDualCSAEAnalyzer:
    """
    Analyzes Dual ConvSAE feature activation patterns across images from the same class.
    Separately analyzes shared pathway (global features) and class pathway (discriminative features).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dual_csae_model: DualConvSAE,
        device: torch.device,
        class_names: List[str],
        dataset: datasets.ImageFolder,
        use_resnet18: bool = False,
        target_layer=None
    ):
        self.device = device
        self.use_resnet18 = use_resnet18

        if use_resnet18:
            # Use ResNet18 backbone
            self.model = ResNet18Wrapper(device=device)
            self.target_layer = self.model.feature_extractor
        else:
            # Use AlexNet backbone
            self.model = model.to(device)
            self.model.eval()
            if target_layer is None:
                self.target_layer = model.feature_extractor[5]
            else:
                self.target_layer = target_layer

        self.dual_csae_model = dual_csae_model.to(device)
        self.dual_csae_model.eval()
        self.class_names = class_names
        self.dataset = dataset
        self.results_by_class = {}

        # GradCAM
        self.gradcam = GradCAM(self.model, self.target_layer)

        # Preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def apply_robust_normalization(self, activation_map: torch.Tensor) -> torch.Tensor:
        """Apply robust normalization (same as training)."""
        scale_factor = torch.quantile(activation_map.flatten(), 0.99)
        normalized = torch.clamp(activation_map, min=0.0, max=scale_factor)
        normalized = normalized / (scale_factor + 1e-8)
        return normalized

    def get_dual_features_for_image(
        self,
        image: torch.Tensor,
        class_idx: int,
        top_k: int = 50,
        cumulative_threshold: float = 0.8
    ) -> Dict:
        """
        Get top activated features from both shared and class-specific pathways.

        Returns:
            Dictionary with:
                - 'shared_feature_indices': List of activated shared feature indices
                - 'shared_feature_scores': Corresponding activation scores
                - 'class_feature_indices': List of activated class feature indices
                - 'class_feature_scores': Corresponding activation scores
                - 'num_channels_analyzed': Number of activation channels analyzed
        """
        image = image.to(self.device)

        # Get GradCAM channel weights
        if self.use_resnet18:
            # Map to ImageNet class for ResNet18
            imagenet_class_idx = IMAGENETTE_TO_IMAGENET[self.class_names[class_idx]]
            channel_weights, cam_map, _ = self.gradcam.forward(image, imagenet_class_idx)
        else:
            channel_weights, cam_map, _ = self.gradcam.forward(image, class_idx)

        activation_maps = self.gradcam.activations  # [1, C, H, W]

        # Select top channels using cumulative threshold
        abs_weights = channel_weights.abs()
        sorted_indices = torch.argsort(abs_weights, descending=True)
        sorted_weights = abs_weights[sorted_indices]

        total_weight = sorted_weights.sum()
        cumulative_sum = torch.cumsum(sorted_weights, dim=0)
        cumulative_ratio = cumulative_sum / total_weight

        n_selected = (cumulative_ratio <= cumulative_threshold).sum().item() + 1
        n_selected = min(n_selected, len(sorted_indices))
        selected_channels = sorted_indices[:n_selected].tolist()

        # Aggregate features from both pathways across all selected channels
        shared_feature_scores = defaultdict(float)
        class_feature_scores = defaultdict(float)

        for ch_idx in selected_channels:
            # Extract and normalize activation map
            act_map = activation_maps[0:1, ch_idx:ch_idx+1, :, :]  # [1, 1, H, W]
            act_map_norm = self.apply_robust_normalization(act_map)

            # Pass through Dual CSAE
            with torch.no_grad():
                reconstruction, shared_features, class_features, _ = self.dual_csae_model(act_map_norm)

            # Get feature importance for this channel
            # shared_features: [1, shared_dim, H, W]
            # class_features: [1, class_dim, H, W]
            shared_importance = shared_features.sum(dim=(2, 3)).squeeze()  # [shared_dim]
            class_importance = class_features.sum(dim=(2, 3)).squeeze()    # [class_dim]

            # Accumulate scores
            for feat_idx in range(shared_importance.shape[0]):
                score = shared_importance[feat_idx].item()
                if score > 0:
                    shared_feature_scores[feat_idx] += score

            for feat_idx in range(class_importance.shape[0]):
                score = class_importance[feat_idx].item()
                if score > 0:
                    class_feature_scores[feat_idx] += score

        # Get top-k features from each pathway
        sorted_shared = sorted(
            shared_feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        sorted_class = sorted(
            class_feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return {
            'shared_feature_indices': [idx for idx, _ in sorted_shared],
            'shared_feature_scores': [score for _, score in sorted_shared],
            'class_feature_indices': [idx for idx, _ in sorted_class],
            'class_feature_scores': [score for _, score in sorted_class],
            'num_channels_analyzed': len(selected_channels)
        }

    def analyze_class(
        self,
        class_name: str,
        num_images: int = 10,
        top_k_features: int = 50,
        cumulative_threshold: float = 0.8
    ) -> Dict:
        """
        Analyze Dual CSAE feature activations for multiple images from the same class.

        Returns:
            analysis_results: Dictionary containing results for both pathways
        """
        # Get class index
        if class_name not in self.class_names:
            raise ValueError(f"Class '{class_name}' not found. Available: {self.class_names}")

        class_idx = self.class_names.index(class_name)
        print(f"\n{'='*70}")
        print(f"Analyzing class: {class_name} (index: {class_idx})")
        print(f"{'='*70}\n")

        # Get indices of images belonging to this class
        class_indices = [i for i, (_, label) in enumerate(self.dataset.samples) if label == class_idx]

        if len(class_indices) == 0:
            raise ValueError(f"No images found for class '{class_name}'")

        # Sample images
        num_samples = min(num_images, len(class_indices))
        sampled_indices = np.random.choice(class_indices, num_samples, replace=False)

        print(f"Analyzing {num_samples} images from class '{class_name}'...")

        # Analyze each image
        image_results = []
        shared_freq = defaultdict(int)
        shared_importance = defaultdict(list)
        class_freq = defaultdict(int)
        class_importance = defaultdict(list)

        for idx, img_idx in enumerate(sampled_indices):
            img_path, label = self.dataset.samples[img_idx]
            image = self.dataset[img_idx][0].unsqueeze(0)  # Add batch dimension

            print(f"[{idx+1}/{num_samples}] Processing image {img_idx}...")

            # Get top features for this image
            result = self.get_dual_features_for_image(
                image,
                class_idx,
                top_k=top_k_features,
                cumulative_threshold=cumulative_threshold
            )

            result['image_path'] = img_path
            result['image_idx'] = img_idx
            image_results.append(result)

            # Update frequency and importance for SHARED pathway
            for feat_idx, score in zip(result['shared_feature_indices'], result['shared_feature_scores']):
                shared_freq[feat_idx] += 1
                shared_importance[feat_idx].append(score)

            # Update frequency and importance for CLASS pathway
            for feat_idx, score in zip(result['class_feature_indices'], result['class_feature_scores']):
                class_freq[feat_idx] += 1
                class_importance[feat_idx].append(score)

        # Compute average importance
        shared_avg_importance = {
            feat_idx: np.mean(scores)
            for feat_idx, scores in shared_importance.items()
        }
        class_avg_importance = {
            feat_idx: np.mean(scores)
            for feat_idx, scores in class_importance.items()
        }

        # Sort features by frequency
        top_shared_features = sorted(
            shared_freq.items(),
            key=lambda x: (x[1], shared_avg_importance.get(x[0], 0)),
            reverse=True
        )

        top_class_features = sorted(
            class_freq.items(),
            key=lambda x: (x[1], class_avg_importance.get(x[0], 0)),
            reverse=True
        )

        # Store results
        results = {
            'class_name': class_name,
            'class_idx': class_idx,
            'num_images': num_samples,
            'image_results': image_results,
            # Shared pathway
            'shared_feature_frequency': dict(shared_freq),
            'shared_feature_avg_importance': shared_avg_importance,
            'top_shared_features': top_shared_features[:50],
            # Class pathway
            'class_feature_frequency': dict(class_freq),
            'class_feature_avg_importance': class_avg_importance,
            'top_class_features': top_class_features[:50]
        }

        self.results_by_class[class_name] = results

        # Print summary
        print(f"\n{'='*70}")
        print(f"Summary for class '{class_name}':")
        print(f"{'='*70}")
        print(f"\nSHARED PATHWAY (Global Features):")
        print(f"  Total unique features: {len(shared_freq)}")
        print(f"  Top 5 most shared features:")
        for i, (feat_idx, freq) in enumerate(top_shared_features[:5], 1):
            avg_imp = shared_avg_importance[feat_idx]
            pct = (freq / num_samples) * 100
            print(f"    {i}. Feature S{feat_idx}: {freq}/{num_samples} ({pct:.1f}%), avg: {avg_imp:.2f}")

        print(f"\nCLASS PATHWAY (Discriminative Features):")
        print(f"  Total unique features: {len(class_freq)}")
        print(f"  Top 5 most shared features:")
        for i, (feat_idx, freq) in enumerate(top_class_features[:5], 1):
            avg_imp = class_avg_importance[feat_idx]
            pct = (freq / num_samples) * 100
            print(f"    {i}. Feature C{feat_idx}: {freq}/{num_samples} ({pct:.1f}%), avg: {avg_imp:.2f}")

        return results

    def visualize_single_class(self, class_name: str, save_dir: str = 'same_class_dual_csae_analysis'):
        """
        Visualize Dual CSAE feature activation patterns for a single class.
        Shows both shared and class-specific pathways.
        """
        if class_name not in self.results_by_class:
            raise ValueError(f"No results found for class '{class_name}'. Run analyze_class first.")

        results = self.results_by_class[class_name]
        os.makedirs(save_dir, exist_ok=True)

        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)

        # Title
        backbone = "ResNet18" if self.use_resnet18 else "AlexNet"
        fig.suptitle(
            f"Dual ConvSAE Analysis ({backbone}): {class_name} ({results['num_images']} images)",
            fontsize=16, fontweight='bold'
        )

        # ===== ROW 1: Sample Images + Top Features Bar Charts =====

        # Sample images (left)
        ax_imgs = fig.add_subplot(gs[0, 0])
        n_show = min(9, len(results['image_results']))
        img_grid = []
        for i in range(n_show):
            img_path = results['image_results'][i]['image_path']
            from PIL import Image
            img = Image.open(img_path).convert('RGB').resize((64, 64))
            img_grid.append(np.array(img))

        # Arrange in 3x3 grid
        rows = []
        for i in range(0, n_show, 3):
            row_imgs = img_grid[i:i+3]
            while len(row_imgs) < 3:
                row_imgs.append(np.zeros_like(img_grid[0]))
            rows.append(np.hstack(row_imgs))
        grid = np.vstack(rows)

        ax_imgs.imshow(grid)
        ax_imgs.set_title(f"Sample Images (n={n_show})", fontsize=11, fontweight='bold')
        ax_imgs.axis('off')

        # Shared features bar chart (middle-left)
        ax_shared_bar = fig.add_subplot(gs[0, 1])
        top_shared = results['top_shared_features'][:15]
        shared_ids = [f"S{idx}" for idx, _ in top_shared]
        shared_freqs = [freq for _, freq in top_shared]
        colors_shared = plt.cm.Greens(np.array(shared_freqs) / max(shared_freqs + [1]))

        ax_shared_bar.barh(range(len(shared_ids)), shared_freqs, color=colors_shared)
        ax_shared_bar.set_yticks(range(len(shared_ids)))
        ax_shared_bar.set_yticklabels(shared_ids, fontsize=7)
        ax_shared_bar.set_xlabel('Frequency', fontsize=9)
        ax_shared_bar.set_title('Top 15 Shared Features', fontsize=11, fontweight='bold', color='green')
        ax_shared_bar.invert_yaxis()
        ax_shared_bar.grid(axis='x', alpha=0.3)

        # Add percentage labels
        for i, freq in enumerate(shared_freqs):
            pct = (freq / results['num_images']) * 100
            ax_shared_bar.text(freq + 0.1, i, f'{pct:.0f}%', va='center', fontsize=6)

        # Class features bar chart (middle-right)
        ax_class_bar = fig.add_subplot(gs[0, 2:])
        top_class = results['top_class_features'][:15]
        class_ids = [f"C{idx}" for idx, _ in top_class]
        class_freqs = [freq for _, freq in top_class]
        colors_class = plt.cm.Oranges(np.array(class_freqs) / max(class_freqs + [1]))

        ax_class_bar.barh(range(len(class_ids)), class_freqs, color=colors_class)
        ax_class_bar.set_yticks(range(len(class_ids)))
        ax_class_bar.set_yticklabels(class_ids, fontsize=7)
        ax_class_bar.set_xlabel('Frequency', fontsize=9)
        ax_class_bar.set_title('Top 15 Class-Specific Features', fontsize=11, fontweight='bold', color='orange')
        ax_class_bar.invert_yaxis()
        ax_class_bar.grid(axis='x', alpha=0.3)

        # Add percentage labels
        for i, freq in enumerate(class_freqs):
            pct = (freq / results['num_images']) * 100
            ax_class_bar.text(freq + 0.1, i, f'{pct:.0f}%', va='center', fontsize=6)

        # ===== ROW 2-3: Feature Activation Heatmaps =====

        # Shared pathway heatmap
        ax_shared_heat = fig.add_subplot(gs[1:3, 0:2])
        top_n_shared = 40
        top_shared_indices = [idx for idx, _ in results['top_shared_features'][:top_n_shared]]
        n_images = len(results['image_results'])

        shared_matrix = np.zeros((n_images, top_n_shared))
        for i, img_result in enumerate(results['image_results']):
            for feat_idx, score in zip(img_result['shared_feature_indices'], img_result['shared_feature_scores']):
                if feat_idx in top_shared_indices:
                    col_idx = top_shared_indices.index(feat_idx)
                    shared_matrix[i, col_idx] = score

        # Normalize
        row_max = shared_matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        shared_matrix_norm = shared_matrix / row_max

        sns.heatmap(
            shared_matrix_norm,
            cmap='Greens',
            cbar_kws={'label': 'Normalized Activation'},
            xticklabels=[f"S{idx}" for idx in top_shared_indices],
            yticklabels=[f"Img {i+1}" for i in range(n_images)],
            ax=ax_shared_heat
        )
        ax_shared_heat.set_xlabel('Shared Feature Index', fontsize=10)
        ax_shared_heat.set_ylabel('Image Index', fontsize=10)
        ax_shared_heat.set_title(
            f'Shared Features Heatmap (Top {top_n_shared})',
            fontsize=11, fontweight='bold', color='green'
        )
        ax_shared_heat.set_xticklabels(ax_shared_heat.get_xticklabels(), rotation=90, fontsize=6)
        ax_shared_heat.set_yticklabels(ax_shared_heat.get_yticklabels(), rotation=0, fontsize=7)

        # Class pathway heatmap
        ax_class_heat = fig.add_subplot(gs[1:3, 2:])
        top_n_class = 40
        top_class_indices = [idx for idx, _ in results['top_class_features'][:top_n_class]]

        class_matrix = np.zeros((n_images, top_n_class))
        for i, img_result in enumerate(results['image_results']):
            for feat_idx, score in zip(img_result['class_feature_indices'], img_result['class_feature_scores']):
                if feat_idx in top_class_indices:
                    col_idx = top_class_indices.index(feat_idx)
                    class_matrix[i, col_idx] = score

        # Normalize
        row_max = class_matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        class_matrix_norm = class_matrix / row_max

        sns.heatmap(
            class_matrix_norm,
            cmap='Oranges',
            cbar_kws={'label': 'Normalized Activation'},
            xticklabels=[f"C{idx}" for idx in top_class_indices],
            yticklabels=[f"Img {i+1}" for i in range(n_images)],
            ax=ax_class_heat
        )
        ax_class_heat.set_xlabel('Class Feature Index', fontsize=10)
        ax_class_heat.set_ylabel('Image Index', fontsize=10)
        ax_class_heat.set_title(
            f'Class Features Heatmap (Top {top_n_class})',
            fontsize=11, fontweight='bold', color='orange'
        )
        ax_class_heat.set_xticklabels(ax_class_heat.get_xticklabels(), rotation=90, fontsize=6)
        ax_class_heat.set_yticklabels(ax_class_heat.get_yticklabels(), rotation=0, fontsize=7)

        # ===== ROW 4: Decoder Weight Distributions =====

        # Shared decoder weights
        ax_shared_weights = fig.add_subplot(gs[3, 0:2])
        shared_decoder_weights = self.dual_csae_model.shared_decoder.weight.detach().cpu().flatten().numpy()
        top_shared_weights = [shared_decoder_weights[idx] for idx, _ in results['top_shared_features'][:40]]

        ax_shared_weights.hist(top_shared_weights, bins=25, color='green', alpha=0.6, edgecolor='black')
        ax_shared_weights.axvline(np.mean(top_shared_weights), color='darkgreen', linestyle='--',
                                   linewidth=2, label=f'Mean: {np.mean(top_shared_weights):.3f}')
        ax_shared_weights.set_xlabel('Decoder Weight', fontsize=9)
        ax_shared_weights.set_ylabel('Count', fontsize=9)
        ax_shared_weights.set_title('Shared Decoder Weights (Top 40 Features)', fontsize=10, fontweight='bold')
        ax_shared_weights.legend(fontsize=8)
        ax_shared_weights.grid(axis='y', alpha=0.3)

        # Class decoder weights
        ax_class_weights = fig.add_subplot(gs[3, 2:])
        class_decoder_weights = self.dual_csae_model.class_decoder.weight.detach().cpu().flatten().numpy()
        top_class_weights = [class_decoder_weights[idx] for idx, _ in results['top_class_features'][:40]]

        ax_class_weights.hist(top_class_weights, bins=25, color='orange', alpha=0.6, edgecolor='black')
        ax_class_weights.axvline(np.mean(top_class_weights), color='darkorange', linestyle='--',
                                  linewidth=2, label=f'Mean: {np.mean(top_class_weights):.3f}')
        ax_class_weights.set_xlabel('Decoder Weight', fontsize=9)
        ax_class_weights.set_ylabel('Count', fontsize=9)
        ax_class_weights.set_title('Class Decoder Weights (Top 40 Features)', fontsize=10, fontweight='bold')
        ax_class_weights.legend(fontsize=8)
        ax_class_weights.grid(axis='y', alpha=0.3)

        # Save
        save_path = os.path.join(save_dir, f'{class_name}_dual_csae_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")
        plt.close()

    def compare_classes(self, class_names: List[str], save_dir: str = 'same_class_dual_csae_analysis'):
        """
        Compare Dual CSAE feature activation patterns across multiple classes.
        Shows both shared and class-specific pathways.
        """
        if not all(name in self.results_by_class for name in class_names):
            missing = [name for name in class_names if name not in self.results_by_class]
            raise ValueError(f"Missing results for classes: {missing}. Run analyze_class first.")

        os.makedirs(save_dir, exist_ok=True)

        # Build feature-class matrices for BOTH pathways
        all_shared_features = set()
        all_class_features = set()

        for class_name in class_names:
            all_shared_features.update(self.results_by_class[class_name]['shared_feature_frequency'].keys())
            all_class_features.update(self.results_by_class[class_name]['class_feature_frequency'].keys())

        all_shared_features = sorted(list(all_shared_features))
        all_class_features = sorted(list(all_class_features))

        n_shared = len(all_shared_features)
        n_class_feats = len(all_class_features)
        n_classes = len(class_names)

        # Matrix: rows = features, cols = classes
        shared_matrix = np.zeros((n_shared, n_classes))
        class_matrix = np.zeros((n_class_feats, n_classes))

        for j, class_name in enumerate(class_names):
            results = self.results_by_class[class_name]
            total_images = results['num_images']

            # Shared features
            for i, feat_idx in enumerate(all_shared_features):
                freq = results['shared_feature_frequency'].get(feat_idx, 0)
                shared_matrix[i, j] = freq / total_images

            # Class features
            for i, feat_idx in enumerate(all_class_features):
                freq = results['class_feature_frequency'].get(feat_idx, 0)
                class_matrix[i, j] = freq / total_images

        # Find class-specific features for CLASS pathway
        class_specificity = []
        for i, feat_idx in enumerate(all_class_features):
            row = class_matrix[i, :]
            max_val = row.max()
            mean_val = row.mean()
            specificity_score = max_val - mean_val

            if max_val > 0.5:
                dominant_class_idx = row.argmax()
                class_specificity.append((feat_idx, specificity_score, dominant_class_idx, max_val))

        class_specificity.sort(key=lambda x: x[1], reverse=True)

        # Visualization
        fig = plt.figure(figsize=(22, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        backbone = "ResNet18" if self.use_resnet18 else "AlexNet"
        fig.suptitle(
            f'Dual ConvSAE ({backbone}): Cross-Class Feature Comparison',
            fontsize=16, fontweight='bold'
        )

        # ===== TOP ROW: Shared Features =====

        # Shared features heatmap
        ax_shared_heat = fig.add_subplot(gs[0, 0:2])
        top_shared_to_show = 60
        shared_subset = shared_matrix[:top_shared_to_show, :]

        sns.heatmap(
            shared_subset,
            cmap='Greens',
            cbar_kws={'label': 'Activation Frequency'},
            xticklabels=class_names,
            yticklabels=[f"S{idx}" for idx in all_shared_features[:top_shared_to_show]],
            ax=ax_shared_heat
        )
        ax_shared_heat.set_xlabel('Class', fontsize=11)
        ax_shared_heat.set_ylabel('Shared Feature Index', fontsize=11)
        ax_shared_heat.set_title(
            f'Shared Features Across Classes (Top {top_shared_to_show})',
            fontsize=12, fontweight='bold', color='green'
        )
        ax_shared_heat.set_yticklabels(ax_shared_heat.get_yticklabels(), fontsize=5)
        ax_shared_heat.set_xticklabels(ax_shared_heat.get_xticklabels(), rotation=45, fontsize=9)

        # Shared features - per-class activation summary
        ax_shared_summary = fig.add_subplot(gs[0, 2])
        shared_per_class = shared_matrix.mean(axis=0)  # Average activation per class
        colors = plt.cm.Greens(shared_per_class / (shared_per_class.max() + 1e-8))

        ax_shared_summary.barh(range(n_classes), shared_per_class, color=colors)
        ax_shared_summary.set_yticks(range(n_classes))
        ax_shared_summary.set_yticklabels(class_names, fontsize=9)
        ax_shared_summary.set_xlabel('Avg Activation', fontsize=10)
        ax_shared_summary.set_title('Shared Features\nPer-Class Average', fontsize=11, fontweight='bold', color='green')
        ax_shared_summary.invert_yaxis()
        ax_shared_summary.grid(axis='x', alpha=0.3)

        # ===== BOTTOM ROW: Class-Specific Features =====

        # Class features heatmap
        ax_class_heat = fig.add_subplot(gs[1, 0:2])
        top_class_to_show = 60
        class_subset = class_matrix[:top_class_to_show, :]

        sns.heatmap(
            class_subset,
            cmap='Oranges',
            cbar_kws={'label': 'Activation Frequency'},
            xticklabels=class_names,
            yticklabels=[f"C{idx}" for idx in all_class_features[:top_class_to_show]],
            ax=ax_class_heat
        )
        ax_class_heat.set_xlabel('Class', fontsize=11)
        ax_class_heat.set_ylabel('Class Feature Index', fontsize=11)
        ax_class_heat.set_title(
            f'Class-Discriminative Features Across Classes (Top {top_class_to_show})',
            fontsize=12, fontweight='bold', color='orange'
        )
        ax_class_heat.set_yticklabels(ax_class_heat.get_yticklabels(), fontsize=5)
        ax_class_heat.set_xticklabels(ax_class_heat.get_xticklabels(), rotation=45, fontsize=9)

        # Class-specific features bar chart
        ax_class_specific = fig.add_subplot(gs[1, 2])
        top_specific = class_specificity[:12]

        feature_labels = [f"C{feat_idx}" for feat_idx, _, _, _ in top_specific]
        specificity_scores = [score for _, score, _, _ in top_specific]
        dominant_classes = [class_names[cls_idx] for _, _, cls_idx, _ in top_specific]

        colors = [plt.cm.tab10(i % 10) for i in [class_names.index(cls) for cls in dominant_classes]]

        ax_class_specific.barh(range(len(feature_labels)), specificity_scores, color=colors)
        ax_class_specific.set_yticks(range(len(feature_labels)))
        ax_class_specific.set_yticklabels(
            [f"{label}\n({cls})" for label, cls in zip(feature_labels, dominant_classes)],
            fontsize=7
        )
        ax_class_specific.set_xlabel('Specificity Score', fontsize=10)
        ax_class_specific.set_title('Top 12 Class-Specific\nFeatures', fontsize=11, fontweight='bold', color='orange')
        ax_class_specific.invert_yaxis()
        ax_class_specific.grid(axis='x', alpha=0.3)

        # Save
        save_path = os.path.join(save_dir, 'class_comparison_dual_csae.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison visualization saved to {save_path}")
        plt.close()

        # Print summary
        print(f"\n{'='*70}")
        print("Class-Specific Features Summary:")
        print(f"{'='*70}")
        for feat_idx, score, cls_idx, freq in top_specific[:10]:
            print(f"Feature C{feat_idx}: {class_names[cls_idx]} "
                  f"({freq*100:.1f}% frequency, specificity: {score:.2f})")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Dual ConvSAE (shared + class-specific) feature consistency'
    )
    parser.add_argument('--class_name', type=str, help='Single class to analyze')
    parser.add_argument('--compare_classes', nargs='+', help='Multiple classes to compare')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images per class')
    parser.add_argument('--top_k_features', type=int, default=50, help='Number of top features to track')
    parser.add_argument('--cumulative_threshold', type=float, default=0.8,
                       help='Cumulative threshold for channel selection')
    parser.add_argument('--use_resnet18', action='store_true',
                       help='Use ResNet18 backbone (for dual_csae_resnet18_model.pkl)')
    parser.add_argument('--model_path', type=str, default='weights/finetune_weights.pth',
                       help='Path to fine-tuned model (ignored if --use_resnet18)')
    parser.add_argument('--dual_csae_path', type=str, default='dual_csae_model.pkl',
                       help='Path to dual CSAE model')
    parser.add_argument('--data_dir', type=str, default='data/imagenette')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    if args.use_resnet18:
        print("Using ResNet18 backbone (pretrained ImageNet)...")
        model = None  # Will be created in analyzer
    else:
        print("Loading AlexNet-based model...")
        model = FineTunedModel(num_classes=10).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

    # Load Dual CSAE
    print(f"Loading Dual ConvSAE from {args.dual_csae_path}...")
    dual_csae_model = joblib.load(args.dual_csae_path)
    dual_csae_model = dual_csae_model.to(device)
    dual_csae_model.eval()

    # Load dataset
    print("Loading dataset...")
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=args.data_dir, transform=data_transform)
    class_names = [d.name for d in os.scandir(args.data_dir) if d.is_dir()]
    class_names.sort()

    print(f"Found {len(class_names)} classes: {class_names}")

    # Create analyzer
    analyzer = SameClassDualCSAEAnalyzer(
        model=model,
        dual_csae_model=dual_csae_model,
        device=device,
        class_names=class_names,
        dataset=dataset,
        use_resnet18=args.use_resnet18
    )

    # Analyze classes
    if args.compare_classes:
        # Compare multiple classes
        print(f"\nComparing classes: {args.compare_classes}")
        for class_name in args.compare_classes:
            analyzer.analyze_class(
                class_name=class_name,
                num_images=args.num_images,
                top_k_features=args.top_k_features,
                cumulative_threshold=args.cumulative_threshold
            )
            analyzer.visualize_single_class(class_name)

        # Generate comparison visualization
        analyzer.compare_classes(args.compare_classes)

    elif args.class_name:
        # Analyze single class
        analyzer.analyze_class(
            class_name=args.class_name,
            num_images=args.num_images,
            top_k_features=args.top_k_features,
            cumulative_threshold=args.cumulative_threshold
        )
        analyzer.visualize_single_class(args.class_name)

    else:
        parser.error("Must provide either --class_name or --compare_classes")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
