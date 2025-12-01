"""
Same-Class ConvSAE Feature Activation Analysis

This script analyzes whether images from the same class share similar activated ConvSAE features.
It helps validate that the learned CSAE has captured semantically meaningful patterns.

Usage:
    # Analyze a single class
    python check_same_class_csae.py --class_name tench --num_images 10

    # Compare multiple classes
    python check_same_class_csae.py --compare_classes tench church parachute --num_images 10
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from torchvision import datasets, transforms
from collections import defaultdict
from typing import Dict, List
import argparse
import os
import joblib

from src.model import FineTunedModel
from src.convsae import ConvSAE
from src.gradcam import GradCAM


class SameClassCSAEAnalyzer:
    """
    Analyzes CSAE feature activation patterns across images from the same class.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        csae_model: ConvSAE,
        device: torch.device,
        class_names: List[str],
        dataset: datasets.ImageFolder,
        target_layer=None
    ):
        self.model = model.to(device)
        self.model.eval()
        self.csae_model = csae_model.to(device)
        self.csae_model.eval()
        self.device = device
        self.class_names = class_names
        self.dataset = dataset
        self.results_by_class = {}

        # Target layer
        if target_layer is None:
            self.target_layer = model.feature_extractor[5]
        else:
            self.target_layer = target_layer

        # GradCAM
        self.gradcam = GradCAM(model, self.target_layer)

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
        normalized = normalized * 10
        return normalized

    def get_top_features_for_image(
        self,
        image: torch.Tensor,
        class_idx: int,
        top_k: int = 50,
        cumulative_threshold: float = 0.8
    ) -> Dict:
        """
        Get top activated CSAE features for a single image.

        Returns:
            Dictionary with:
                - 'feature_indices': List of activated feature indices
                - 'feature_scores': Corresponding activation scores
                - 'num_channels_analyzed': Number of activation channels analyzed
        """
        image = image.to(self.device)

        # Get GradCAM channel weights
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

        # Aggregate CSAE features across all selected channels
        feature_scores_aggregated = defaultdict(float)

        for ch_idx in selected_channels:
            # Extract and normalize activation map
            act_map = activation_maps[0:1, ch_idx:ch_idx+1, :, :]  # [1, 1, H, W]
            act_map_norm = self.apply_robust_normalization(act_map)

            # Pass through CSAE
            with torch.no_grad():
                reconstruction, sparse_features = self.csae_model(act_map_norm)

            # Get feature importance for this channel
            # sparse_features: [1, hidden_dim, H, W]
            feature_importance = sparse_features.sum(dim=(2, 3)).squeeze()  # [hidden_dim]

            # Accumulate scores
            for feat_idx in range(feature_importance.shape[0]):
                score = feature_importance[feat_idx].item()
                if score > 0:  # Only count active features
                    feature_scores_aggregated[feat_idx] += score

        # Get top-k features
        sorted_features = sorted(
            feature_scores_aggregated.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        feature_indices = [idx for idx, _ in sorted_features]
        feature_scores = [score for _, score in sorted_features]

        return {
            'feature_indices': feature_indices,
            'feature_scores': feature_scores,
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
        Analyze CSAE feature activations for multiple images from the same class.

        Returns:
            analysis_results: Dictionary containing:
                - 'class_name': Class name
                - 'class_idx': Class index
                - 'image_results': List of results for each image
                - 'feature_frequency': How many images activate each feature
                - 'feature_importance': Average importance score for each feature
                - 'top_shared_features': Features that appear in most images
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
        feature_frequency = defaultdict(int)  # Count how many images activate each feature
        feature_importance = defaultdict(list)  # Track importance scores for each feature

        for idx, img_idx in enumerate(sampled_indices):
            img_path, label = self.dataset.samples[img_idx]
            image = self.dataset[img_idx][0].unsqueeze(0)  # Add batch dimension

            print(f"[{idx+1}/{num_samples}] Processing image {img_idx}...")

            # Get top features for this image
            result = self.get_top_features_for_image(
                image,
                class_idx,
                top_k=top_k_features,
                cumulative_threshold=cumulative_threshold
            )

            result['image_path'] = img_path
            result['image_idx'] = img_idx
            image_results.append(result)

            # Update frequency and importance
            for feat_idx, score in zip(result['feature_indices'], result['feature_scores']):
                feature_frequency[feat_idx] += 1
                feature_importance[feat_idx].append(score)

        # Compute average importance
        feature_avg_importance = {
            feat_idx: np.mean(scores)
            for feat_idx, scores in feature_importance.items()
        }

        # Sort features by frequency (how many images activate them)
        top_shared_features = sorted(
            feature_frequency.items(),
            key=lambda x: (x[1], feature_avg_importance.get(x[0], 0)),  # Sort by frequency, then importance
            reverse=True
        )

        # Store results
        results = {
            'class_name': class_name,
            'class_idx': class_idx,
            'num_images': num_samples,
            'image_results': image_results,
            'feature_frequency': dict(feature_frequency),
            'feature_avg_importance': feature_avg_importance,
            'top_shared_features': top_shared_features[:50]  # Top 50 most shared
        }

        self.results_by_class[class_name] = results

        # Print summary
        print(f"\n{'='*70}")
        print(f"Summary for class '{class_name}':")
        print(f"{'='*70}")
        print(f"Total unique features activated: {len(feature_frequency)}")
        print(f"Top 10 most shared features (frequency, avg importance):")
        for i, (feat_idx, freq) in enumerate(top_shared_features[:10], 1):
            avg_imp = feature_avg_importance[feat_idx]
            pct = (freq / num_samples) * 100
            print(f"  {i}. Feature {feat_idx}: {freq}/{num_samples} images ({pct:.1f}%), "
                  f"avg importance: {avg_imp:.2f}")

        # Feature consistency metric
        high_freq_features = [f for f, freq in top_shared_features if freq >= num_samples * 0.7]
        print(f"\nConsistency metric:")
        print(f"  Features appearing in ≥70% of images: {len(high_freq_features)}")
        if high_freq_features:
            print(f"  Indices: {high_freq_features[:10]}{'...' if len(high_freq_features) > 10 else ''}")

        return results

    def visualize_single_class(self, class_name: str, save_dir: str = 'same_class_csae_analysis'):
        """
        Visualize CSAE feature activation patterns for a single class.
        """
        if class_name not in self.results_by_class:
            raise ValueError(f"No results found for class '{class_name}'. Run analyze_class first.")

        results = self.results_by_class[class_name]
        os.makedirs(save_dir, exist_ok=True)

        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

        # Title
        fig.suptitle(
            f"ConvSAE Feature Activation Analysis: {class_name} ({results['num_images']} images)",
            fontsize=16, fontweight='bold'
        )

        # 1. Sample images (top left)
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
        ax_imgs.set_title(f"Sample Images (n={n_show})", fontsize=12, fontweight='bold')
        ax_imgs.axis('off')

        # 2. Top shared features (bar chart)
        ax_freq = fig.add_subplot(gs[0, 1:])
        top_features = results['top_shared_features'][:20]
        feature_ids = [f"F{idx}" for idx, _ in top_features]
        frequencies = [freq for _, freq in top_features]
        colors = plt.cm.viridis(np.array(frequencies) / max(frequencies))

        bars = ax_freq.barh(range(len(feature_ids)), frequencies, color=colors)
        ax_freq.set_yticks(range(len(feature_ids)))
        ax_freq.set_yticklabels(feature_ids, fontsize=8)
        ax_freq.set_xlabel('Frequency (# of images activating this feature)', fontsize=10)
        ax_freq.set_title(f'Top 20 Most Frequently Activated Features', fontsize=12, fontweight='bold')
        ax_freq.invert_yaxis()
        ax_freq.grid(axis='x', alpha=0.3)

        # Add percentage labels
        total_images = results['num_images']
        for i, (bar, freq) in enumerate(zip(bars, frequencies)):
            pct = (freq / total_images) * 100
            ax_freq.text(freq + 0.2, i, f'{pct:.0f}%', va='center', fontsize=8)

        # 3. Feature activation heatmap (per image)
        ax_heatmap = fig.add_subplot(gs[1:3, :])

        # Build matrix: rows = images, cols = top features
        top_n_features = 50
        top_feature_indices = [idx for idx, _ in results['top_shared_features'][:top_n_features]]
        n_images = len(results['image_results'])

        activation_matrix = np.zeros((n_images, top_n_features))

        for i, img_result in enumerate(results['image_results']):
            for feat_idx, score in zip(img_result['feature_indices'], img_result['feature_scores']):
                if feat_idx in top_feature_indices:
                    col_idx = top_feature_indices.index(feat_idx)
                    activation_matrix[i, col_idx] = score

        # Normalize each row for better visualization
        row_max = activation_matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1  # Avoid division by zero
        activation_matrix_norm = activation_matrix / row_max

        # Plot heatmap
        sns.heatmap(
            activation_matrix_norm,
            cmap='YlOrRd',
            cbar_kws={'label': 'Normalized Activation (per image)'},
            xticklabels=[f"F{idx}" for idx in top_feature_indices],
            yticklabels=[f"Img {i+1}" for i in range(n_images)],
            ax=ax_heatmap
        )
        ax_heatmap.set_xlabel('CSAE Feature Index', fontsize=11)
        ax_heatmap.set_ylabel('Image Index', fontsize=11)
        ax_heatmap.set_title(
            f'Feature Activation Heatmap (Top {top_n_features} features)',
            fontsize=12, fontweight='bold'
        )

        # Rotate x-axis labels
        ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=90, fontsize=7)
        ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)

        # 4. Raw activation score distribution for top features
        ax_raw_scores = fig.add_subplot(gs[3, 0:2])

        # Collect raw scores (before normalization) for top shared features
        top_20_features = [idx for idx, _ in results['top_shared_features'][:20]]
        raw_scores_per_feature = []
        feature_labels = []

        for feat_idx in top_20_features:
            scores = []
            for img_result in results['image_results']:
                if feat_idx in img_result['feature_indices']:
                    idx_pos = img_result['feature_indices'].index(feat_idx)
                    scores.append(img_result['feature_scores'][idx_pos])

            if scores:  # Only include if feature was activated in at least one image
                raw_scores_per_feature.append(scores)
                feature_labels.append(f"F{feat_idx}")

        # Box plot of raw scores
        bp = ax_raw_scores.boxplot(
            raw_scores_per_feature,
            labels=feature_labels,
            patch_artist=True,
            showfliers=False
        )

        # Color boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        ax_raw_scores.set_xlabel('Feature Index', fontsize=10)
        ax_raw_scores.set_ylabel('Raw Activation Score', fontsize=10)
        ax_raw_scores.set_title('Raw Activation Score Distribution (Top 20 Features)',
                               fontsize=12, fontweight='bold')
        ax_raw_scores.tick_params(axis='x', rotation=45, labelsize=8)
        ax_raw_scores.grid(axis='y', alpha=0.3)

        # 5. CSAE decoder weight distribution for top features
        ax_weights = fig.add_subplot(gs[3, 2])

        # Get decoder weights for all features
        decoder_weights = self.csae_model.decoder.weight.detach().cpu()  # [1, hidden_dim, 1, 1]
        all_weights = decoder_weights.squeeze().numpy()  # [hidden_dim]

        # Get weights for top shared features
        top_feature_weights = [all_weights[idx] for idx, _ in results['top_shared_features'][:50]]

        # Histogram
        ax_weights.hist(top_feature_weights, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax_weights.axvline(np.mean(top_feature_weights), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(top_feature_weights):.3f}')
        ax_weights.axvline(np.median(top_feature_weights), color='orange', linestyle='--',
                          linewidth=2, label=f'Median: {np.median(top_feature_weights):.3f}')

        ax_weights.set_xlabel('Decoder Weight', fontsize=10)
        ax_weights.set_ylabel('Count', fontsize=10)
        ax_weights.set_title('Decoder Weight Distribution\n(Top 50 Features)',
                            fontsize=11, fontweight='bold')
        ax_weights.legend(fontsize=8)
        ax_weights.grid(axis='y', alpha=0.3)

        # Add statistics text
        stats_text = (
            f"Stats (Top 50):\n"
            f"Min: {np.min(top_feature_weights):.3f}\n"
            f"Max: {np.max(top_feature_weights):.3f}\n"
            f"Std: {np.std(top_feature_weights):.3f}"
        )
        ax_weights.text(0.98, 0.98, stats_text, transform=ax_weights.transAxes,
                       fontsize=8, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Save
        save_path = os.path.join(save_dir, f'{class_name}_csae_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")
        plt.close()

    def compare_classes(self, class_names: List[str], save_dir: str = 'same_class_csae_analysis'):
        """
        Compare CSAE feature activation patterns across multiple classes.
        """
        if not all(name in self.results_by_class for name in class_names):
            missing = [name for name in class_names if name not in self.results_by_class]
            raise ValueError(f"Missing results for classes: {missing}. Run analyze_class first.")

        os.makedirs(save_dir, exist_ok=True)

        # Build feature-class activation matrix
        all_features = set()
        for class_name in class_names:
            all_features.update(self.results_by_class[class_name]['feature_frequency'].keys())

        all_features = sorted(list(all_features))
        n_features = len(all_features)
        n_classes = len(class_names)

        # Matrix: rows = features, cols = classes
        activation_matrix = np.zeros((n_features, n_classes))

        for j, class_name in enumerate(class_names):
            results = self.results_by_class[class_name]
            total_images = results['num_images']

            for i, feat_idx in enumerate(all_features):
                freq = results['feature_frequency'].get(feat_idx, 0)
                activation_matrix[i, j] = freq / total_images  # Normalize by number of images

        # Find class-specific features (high in one class, low in others)
        class_specificity = []
        for i, feat_idx in enumerate(all_features):
            row = activation_matrix[i, :]
            max_val = row.max()
            mean_val = row.mean()
            specificity_score = max_val - mean_val  # High when one class dominates

            if max_val > 0.5:  # Feature appears in >50% of images in at least one class
                dominant_class_idx = row.argmax()
                class_specificity.append((feat_idx, specificity_score, dominant_class_idx, max_val))

        # Sort by specificity
        class_specificity.sort(key=lambda x: x[1], reverse=True)

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # 1. Feature-Class Heatmap (top 100 features)
        top_features_to_show = 100
        top_feature_indices = all_features[:top_features_to_show]
        matrix_subset = activation_matrix[:top_features_to_show, :]

        sns.heatmap(
            matrix_subset,
            cmap='YlGnBu',
            cbar_kws={'label': 'Activation Frequency'},
            xticklabels=class_names,
            yticklabels=[f"F{idx}" for idx in top_feature_indices],
            ax=axes[0]
        )
        axes[0].set_xlabel('Class', fontsize=12)
        axes[0].set_ylabel('CSAE Feature Index', fontsize=12)
        axes[0].set_title(f'Feature Activation Across Classes (Top {top_features_to_show} features)',
                         fontsize=13, fontweight='bold')
        axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=6)

        # 2. Class-Specific Features
        ax_specific = axes[1]
        top_specific = class_specificity[:20]

        feature_labels = [f"F{feat_idx}" for feat_idx, _, _, _ in top_specific]
        specificity_scores = [score for _, score, _, _ in top_specific]
        dominant_classes = [class_names[cls_idx] for _, _, cls_idx, _ in top_specific]

        colors = [plt.cm.tab10(i % 10) for i in [class_names.index(cls) for cls in dominant_classes]]

        bars = ax_specific.barh(range(len(feature_labels)), specificity_scores, color=colors)
        ax_specific.set_yticks(range(len(feature_labels)))
        ax_specific.set_yticklabels(
            [f"{label}\n({cls})" for label, cls in zip(feature_labels, dominant_classes)],
            fontsize=8
        )
        ax_specific.set_xlabel('Class Specificity Score', fontsize=11)
        ax_specific.set_title('Top 20 Class-Specific Features', fontsize=13, fontweight='bold')
        ax_specific.invert_yaxis()
        ax_specific.grid(axis='x', alpha=0.3)

        plt.suptitle(
            f'ConvSAE Feature Comparison Across Classes',
            fontsize=16, fontweight='bold', y=0.98
        )
        plt.tight_layout()

        # Save
        save_path = os.path.join(save_dir, 'class_comparison_csae.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison visualization saved to {save_path}")
        plt.close()

        # Print summary
        print(f"\n{'='*70}")
        print("Class-Specific Features Summary:")
        print(f"{'='*70}")
        for feat_idx, score, cls_idx, freq in top_specific[:10]:
            print(f"Feature {feat_idx}: {class_names[cls_idx]} ({freq*100:.1f}% frequency, specificity: {score:.2f})")


def main():
    parser = argparse.ArgumentParser(description='Analyze ConvSAE feature consistency across same-class images')
    parser.add_argument('--class_name', type=str, help='Single class to analyze')
    parser.add_argument('--compare_classes', nargs='+', help='Multiple classes to compare')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images per class')
    parser.add_argument('--top_k_features', type=int, default=50, help='Number of top features to track')
    parser.add_argument('--cumulative_threshold', type=float, default=0.8,
                       help='Cumulative threshold for channel selection')
    parser.add_argument('--model_path', type=str, default='weights/finetune_weights.pth')
    parser.add_argument('--csae_path', type=str, default='csae_model.pkl')
    parser.add_argument('--data_dir', type=str, default='data/imagenette')

    args = parser.parse_args()

    # Setup
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
    analyzer = SameClassCSAEAnalyzer(
        model=model,
        csae_model=csae_model,
        device=device,
        class_names=class_names,
        dataset=dataset
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
