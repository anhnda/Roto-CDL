"""
Multi-Channel ConvSAE Feature Analysis Script

Analyzes the learned sparse features from Multi-Channel ConvSAE to understand:
1. Which input channels activate each feature
2. Feature specialization patterns (channel combinations)
3. Top-k most important features
4. Channel usage distribution

Inspired by SAE feature analysis in LLM interpretability.

Usage:
    # Analyze all features
    python analyze_multichannel_csae.py

    # Analyze specific features
    python analyze_multichannel_csae.py --feature_indices 0 10 42 100
"""

import torch
import torch.nn.functional as F
import joblib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

# Import our model class
import sys
sys.path.append('.')
from run_multichannel_csae_resnet18 import MultiChannelConvSAE


class MultiChannelCSAEAnalyzer:
    """
    Analyzer for Multi-Channel ConvSAE learned features.

    Provides interpretability tools to understand what each learned feature
    represents in terms of input channel combinations.
    """

    def __init__(self, model_path: str = 'multichannel_csae_resnet18_model.pkl', device='cuda'):
        """
        Args:
            model_path: Path to saved ConvSAE model
            device: Device to run analysis on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path).to(self.device)
        self.model.eval()

        print(f"Model loaded successfully!")
        print(f"  Input channels: {self.model.in_channels}")
        print(f"  Hidden dim: {self.model.hidden_dim}")
        print(f"  Kernel size: {self.model.kernel_size}")

    def get_decoder_weights(self) -> torch.Tensor:
        """
        Get decoder weights.

        Returns:
            weights: [in_channels, hidden_dim] - Decoder weight matrix (for 1×1 conv)
        """
        # Decoder weight shape: [in_channels, hidden_dim, kernel_size, kernel_size]
        weights = self.model.decoder.weight.data  # [256, 4096, 1, 1]
        weights = weights.squeeze()  # [256, 4096]

        return weights

    def get_feature_channel_importance(self, feature_idx: int, top_k: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the top-k most important input channels for a specific feature.

        Args:
            feature_idx: Index of the feature to analyze (0 to hidden_dim-1)
            top_k: Number of top channels to return

        Returns:
            channel_indices: [top_k] - Indices of top channels
            channel_weights: [top_k] - Weights of top channels
        """
        weights = self.get_decoder_weights()  # [256, hidden_dim]

        # Get weights for this feature
        feature_weights = weights[:, feature_idx]  # [256]

        # Get top-k by absolute value
        abs_weights = feature_weights.abs()
        top_k_values, top_k_indices = torch.topk(abs_weights, k=min(top_k, len(abs_weights)))

        # Get actual weights (with sign)
        top_k_weights = feature_weights[top_k_indices]

        return top_k_indices.cpu(), top_k_weights.cpu()

    def get_most_important_features(self, top_k: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the top-k most important features (by L2 norm of decoder weights).

        Args:
            top_k: Number of top features to return

        Returns:
            feature_indices: [top_k] - Indices of top features
            feature_norms: [top_k] - L2 norms of top features
        """
        weights = self.get_decoder_weights()  # [256, hidden_dim]

        # Compute L2 norm per feature
        feature_norms = weights.norm(dim=0)  # [hidden_dim]

        # Get top-k
        top_k_norms, top_k_indices = torch.topk(feature_norms, k=min(top_k, len(feature_norms)))

        return top_k_indices.cpu(), top_k_norms.cpu()

    def analyze_feature_specialization(self, threshold: float = 0.1) -> Dict:
        """
        Analyze how specialized each feature is (how many input channels it uses).

        Args:
            threshold: Weight threshold for considering a channel "active"

        Returns:
            stats: Dictionary with specialization statistics
        """
        weights = self.get_decoder_weights()  # [256, hidden_dim]

        # Normalize weights per feature
        weights_norm = weights / (weights.norm(dim=0, keepdim=True) + 1e-8)

        # Count number of "active" channels per feature
        active_channels = (weights_norm.abs() > threshold).float().sum(dim=0)  # [hidden_dim]

        stats = {
            'mean_active_channels': active_channels.mean().item(),
            'std_active_channels': active_channels.std().item(),
            'min_active_channels': active_channels.min().item(),
            'max_active_channels': active_channels.max().item(),
            'median_active_channels': active_channels.median().item(),
            'active_channels_dist': active_channels.cpu().numpy()
        }

        return stats

    def visualize_feature(self, feature_idx: int, top_k: int = 30, save_path: str = None):
        """
        Visualize a specific feature: which input channels it responds to.

        Args:
            feature_idx: Index of feature to visualize
            top_k: Number of top channels to show
            save_path: Path to save visualization (optional)
        """
        channel_indices, channel_weights = self.get_feature_channel_importance(feature_idx, top_k)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Feature {feature_idx} - Input Channel Analysis', fontsize=14, fontweight='bold')

        # 1. Bar chart of top-k channel weights
        colors = ['green' if w > 0 else 'red' for w in channel_weights.numpy()]
        axes[0].bar(range(top_k), channel_weights.numpy(), color=colors, alpha=0.7)
        axes[0].set_title(f'Top {top_k} Channel Weights')
        axes[0].set_xlabel('Rank')
        axes[0].set_ylabel('Weight')
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].grid(True, alpha=0.3)

        # 2. Channel indices
        axes[1].bar(range(top_k), channel_indices.numpy(), color='blue', alpha=0.7)
        axes[1].set_title(f'Top {top_k} Channel Indices (0-255)')
        axes[1].set_xlabel('Rank')
        axes[1].set_ylabel('Channel Index')
        axes[1].set_ylim(0, 256)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Feature visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

        # Print summary
        print(f"\nFeature {feature_idx} Summary:")
        print(f"  Top 5 channels: {channel_indices[:5].tolist()}")
        print(f"  Top 5 weights: {channel_weights[:5].tolist()}")
        print(f"  Weight range: [{channel_weights.min():.4f}, {channel_weights.max():.4f}]")

    def visualize_top_features(self, num_features: int = 16, channels_per_feature: int = 10,
                              save_path: str = 'multichannel_csae_top_features.png'):
        """
        Visualize the top-k most important features in a grid.

        Args:
            num_features: Number of features to visualize
            channels_per_feature: Number of top channels to show per feature
            save_path: Path to save visualization
        """
        # Get top features
        feature_indices, feature_norms = self.get_most_important_features(num_features)

        # Create grid
        ncols = 4
        nrows = (num_features + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
        fig.suptitle(f'Top {num_features} Features by Importance', fontsize=16, fontweight='bold')

        axes = axes.flatten() if num_features > 1 else [axes]

        for i, (feat_idx, feat_norm) in enumerate(zip(feature_indices, feature_norms)):
            channel_indices, channel_weights = self.get_feature_channel_importance(
                feat_idx.item(), channels_per_feature
            )

            # Plot
            colors = ['green' if w > 0 else 'red' for w in channel_weights.numpy()]
            axes[i].bar(range(channels_per_feature), channel_weights.numpy(),
                       color=colors, alpha=0.7)
            axes[i].set_title(f'Feature {feat_idx} (norm: {feat_norm:.3f})', fontsize=10)
            axes[i].set_xlabel('Rank', fontsize=8)
            axes[i].set_ylabel('Weight', fontsize=8)
            axes[i].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(labelsize=8)

            # Add channel indices as text
            top3_channels = channel_indices[:3].tolist()
            axes[i].text(0.95, 0.95, f'Ch: {top3_channels}',
                        transform=axes[i].transAxes,
                        fontsize=7, va='top', ha='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Hide unused subplots
        for i in range(num_features, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Top features visualization saved to {save_path}")
        plt.close()

    def visualize_specialization_stats(self, threshold: float = 0.1,
                                      save_path: str = 'multichannel_csae_specialization.png'):
        """
        Visualize feature specialization statistics.

        Args:
            threshold: Weight threshold for considering a channel "active"
            save_path: Path to save visualization
        """
        stats = self.analyze_feature_specialization(threshold)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Feature Specialization Analysis', fontsize=14, fontweight='bold')

        # 1. Distribution of active channels per feature
        active_dist = stats['active_channels_dist']
        axes[0].hist(active_dist, bins=50, color='purple', alpha=0.7, edgecolor='black')
        axes[0].axvline(stats['mean_active_channels'], color='red', linestyle='--',
                       linewidth=2, label=f"Mean: {stats['mean_active_channels']:.1f}")
        axes[0].axvline(stats['median_active_channels'], color='blue', linestyle='--',
                       linewidth=2, label=f"Median: {stats['median_active_channels']:.1f}")
        axes[0].set_title('Distribution of Active Channels per Feature')
        axes[0].set_xlabel(f'Number of Active Channels (threshold={threshold})')
        axes[0].set_ylabel('Count')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Summary statistics
        axes[1].axis('off')
        summary_text = f"""
        Feature Specialization Statistics
        (Weight threshold: {threshold})

        Mean active channels: {stats['mean_active_channels']:.2f}
        Std dev: {stats['std_active_channels']:.2f}
        Median: {stats['median_active_channels']:.1f}
        Min: {stats['min_active_channels']:.0f}
        Max: {stats['max_active_channels']:.0f}

        Interpretation:
        - Lower values = More specialized features
        - Higher values = More general features

        Target: 20-50 channels per feature
        (sparse combinations of input channels)
        """
        axes[1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                    verticalalignment='center')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Specialization analysis saved to {save_path}")
        plt.close()

        return stats


def main():
    parser = argparse.ArgumentParser(description='Analyze Multi-Channel ConvSAE features')
    parser.add_argument('--model_path', type=str,
                       default='multichannel_csae_resnet18_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--feature_indices', type=int, nargs='+',
                       help='Specific feature indices to analyze')
    parser.add_argument('--num_top_features', type=int, default=16,
                       help='Number of top features to visualize')
    parser.add_argument('--output_dir', type=str, default='multichannel_csae_analysis',
                       help='Output directory for visualizations')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create analyzer
    print("="*80)
    print("Multi-Channel ConvSAE Feature Analysis")
    print("="*80)

    analyzer = MultiChannelCSAEAnalyzer(model_path=args.model_path)

    # ========================================
    # 1. Overall Statistics
    # ========================================
    print("\n" + "="*80)
    print("1. Feature Specialization Analysis")
    print("="*80)

    stats = analyzer.visualize_specialization_stats(
        threshold=0.1,
        save_path=str(output_dir / 'specialization.png')
    )

    print(f"\nFeature Specialization Summary:")
    print(f"  Mean active channels: {stats['mean_active_channels']:.2f}")
    print(f"  Median active channels: {stats['median_active_channels']:.1f}")
    print(f"  Range: [{stats['min_active_channels']:.0f}, {stats['max_active_channels']:.0f}]")

    # ========================================
    # 2. Top Features
    # ========================================
    print("\n" + "="*80)
    print("2. Top Features Visualization")
    print("="*80)

    analyzer.visualize_top_features(
        num_features=args.num_top_features,
        channels_per_feature=10,
        save_path=str(output_dir / 'top_features.png')
    )

    # Get and print top features
    feature_indices, feature_norms = analyzer.get_most_important_features(top_k=10)
    print(f"\nTop 10 Features by Importance:")
    for i, (idx, norm) in enumerate(zip(feature_indices, feature_norms)):
        print(f"  {i+1}. Feature {idx} (L2 norm: {norm:.4f})")

    # ========================================
    # 3. Specific Features (if provided)
    # ========================================
    if args.feature_indices:
        print("\n" + "="*80)
        print("3. Specific Feature Analysis")
        print("="*80)

        for feat_idx in args.feature_indices:
            print(f"\nAnalyzing Feature {feat_idx}...")
            analyzer.visualize_feature(
                feat_idx,
                top_k=30,
                save_path=str(output_dir / f'feature_{feat_idx}.png')
            )

    print("\n" + "="*80)
    print("✓ Analysis complete! Outputs saved to:", output_dir)
    print("="*80)


if __name__ == "__main__":
    main()
