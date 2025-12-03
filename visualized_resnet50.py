"""
Visualization utilities for ResNet50 Multi-Channel ConvSAE (Two-Level Sparsity).

This module provides comprehensive visualization functions for analyzing
the learned features and behavior of the Multi-Channel Convolutional
Sparse Autoencoder trained on ResNet50 layer3 activations.

The model uses two-level sparsity:
- Level 1 (Channel): Top-k channel selection based on sum of spatial activations
- Level 2 (Spatial): L1-sparse activations within selected channels
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def visualize_multichannel_sae_r50(
    model,
    sample_activations: torch.Tensor,
    num_samples: int = 4,
    num_channels_to_show: int = 8,
    num_features_to_show: int = 16,
    save_path: str = 'multichannel_sae_r50_visualization.png'
):
    """
    Comprehensive visualization of ResNet50 Multi-Channel SAE with two-level sparsity.

    The model applies two-level sparsity during encoding:
    - Level 1 (Channel): Ranks features by sum(H×W), keeps top-k channels
    - Level 2 (Spatial): L1-sparse activations within selected channels

    Shows:
    1. Sample input activation maps (selected channels from ResNet50 layer3)
    2. SAE reconstructions
    3. Top activated sparse features (after two-level sparsity)
    4. Feature activation spatial patterns
    5. Channel-wise reconstruction quality

    Args:
        model: Trained MultiChannelConvSAE model with two-level sparsity
        sample_activations: [N, 1024, 14, 14] - Sample activation maps from ResNet50 layer3
        num_samples: Number of sample images to visualize
        num_channels_to_show: Number of input channels to display per sample
        num_features_to_show: Number of top SAE features to display per sample
        save_path: Path to save the visualization
    """
    model.eval()
    device = next(model.parameters()).device

    # Select random samples
    indices = np.random.choice(len(sample_activations), size=min(num_samples, len(sample_activations)), replace=False)
    samples = sample_activations[indices].to(device)

    # Forward pass through SAE
    with torch.no_grad():
        reconstructions, sparse_features = model(samples, use_topk=True)

    # Move to CPU for visualization
    samples = samples.cpu()
    reconstructions = reconstructions.cpu()
    sparse_features = sparse_features.cpu()

    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 5 * num_samples))
    fig.suptitle('ResNet50 Multi-Channel SAE Visualization (Two-Level Sparsity)', fontsize=16, fontweight='bold')

    for i in range(num_samples):
        base_row = i * 4

        # Get data for this sample
        input_act = samples[i]  # [1024, 14, 14]
        recon_act = reconstructions[i]  # [1024, 14, 14]
        sparse_feat = sparse_features[i]  # [hidden_dim, 14, 14]

        # Compute reconstruction error per channel
        recon_error = (input_act - recon_act).pow(2).mean(dim=(1, 2))  # [1024]

        # Find most active channels (non-zero input channels)
        channel_activity = input_act.abs().sum(dim=(1, 2))  # [1024]
        active_channels = torch.nonzero(channel_activity > 1e-6).squeeze()

        # Select top-k active channels to display
        if len(active_channels) > num_channels_to_show:
            _, top_indices = torch.topk(channel_activity[active_channels], num_channels_to_show)
            selected_channels = active_channels[top_indices]
        else:
            selected_channels = active_channels

        # Find top activated sparse features
        feature_activity = sparse_feat.abs().sum(dim=(1, 2))  # [hidden_dim]
        top_feature_vals, top_feature_indices = torch.topk(feature_activity, num_features_to_show)

        # Row 1: Input activation maps (selected channels)
        ax_start = plt.subplot2grid((num_samples * 4, num_channels_to_show + 2),
                                     (base_row, 0), colspan=num_channels_to_show + 2)
        ax_start.text(0.5, 0.5, f'Sample {i+1}\n{len(active_channels)} Active Channels\n'
                               f'MSE: {F.mse_loss(recon_act, input_act).item():.4f}',
                     ha='center', va='center', fontsize=12, fontweight='bold')
        ax_start.axis('off')

        for j, ch_idx in enumerate(selected_channels[:num_channels_to_show]):
            ax = plt.subplot2grid((num_samples * 4, num_channels_to_show + 2),
                                 (base_row + 1, j))
            im = ax.imshow(input_act[ch_idx].numpy(), cmap='viridis', aspect='auto')
            ax.set_title(f'Ch {ch_idx.item()}\nMSE: {recon_error[ch_idx].item():.4f}', fontsize=8)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Row 2: Reconstructed activation maps
        for j, ch_idx in enumerate(selected_channels[:num_channels_to_show]):
            ax = plt.subplot2grid((num_samples * 4, num_channels_to_show + 2),
                                 (base_row + 2, j))
            im = ax.imshow(recon_act[ch_idx].numpy(), cmap='viridis', aspect='auto')
            ax.set_title(f'Recon {ch_idx.item()}', fontsize=8)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Row 3: Top sparse features
        for j in range(min(num_features_to_show, len(top_feature_indices))):
            if j >= num_channels_to_show:
                break
            feat_idx = top_feature_indices[j]
            feat_val = top_feature_vals[j]

            ax = plt.subplot2grid((num_samples * 4, num_channels_to_show + 2),
                                 (base_row + 3, j))
            im = ax.imshow(sparse_feat[feat_idx].numpy(), cmap='hot', aspect='auto')
            ax.set_title(f'Feat {feat_idx.item()}\nAct: {feat_val.item():.2f}', fontsize=8)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Summary statistics on the right
        ax_summary = plt.subplot2grid((num_samples * 4, num_channels_to_show + 2),
                                      (base_row + 1, num_channels_to_show),
                                      rowspan=3, colspan=2)

        # Compute statistics
        sparsity_pct = (sparse_feat > 0).float().mean().item() * 100
        num_active_features = (sparse_feat > 0).any(dim=(1, 2)).sum().item()

        summary_text = f"Sample {i+1} Statistics:\n\n"
        summary_text += f"Input Channels:\n"
        summary_text += f"  Active: {len(active_channels)}/1024\n"
        summary_text += f"  Total activation: {channel_activity.sum().item():.2f}\n\n"
        summary_text += f"Sparse Features:\n"
        summary_text += f"  Active: {num_active_features}/{model.hidden_dim}\n"
        summary_text += f"  Sparsity: {sparsity_pct:.2f}%\n"
        summary_text += f"  Top-K: {model.top_k}\n\n"
        summary_text += f"Reconstruction:\n"
        summary_text += f"  MSE: {F.mse_loss(recon_act, input_act).item():.4f}\n"
        summary_text += f"  MAE: {F.l1_loss(recon_act, input_act).item():.4f}\n"

        ax_summary.text(0.1, 0.5, summary_text, fontsize=9,
                       verticalalignment='center', family='monospace')
        ax_summary.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"ResNet50 Multi-Channel SAE visualization saved to {save_path}")
    plt.close()


def visualize_feature_decoder_weights(
    model,
    num_features: int = 64,
    save_path: str = 'resnet50_decoder_weights.png'
):
    """
    Visualize the decoder weight matrix to understand which ResNet50 channels
    each learned feature uses.

    Args:
        model: Trained MultiChannelConvSAE model
        num_features: Number of top features to visualize
        save_path: Path to save the visualization
    """
    model.eval()

    # Get decoder weights [in_channels=1024, hidden_dim, kernel_size, kernel_size]
    decoder_weights = model.decoder.weight.detach().cpu()
    decoder_weights = decoder_weights.squeeze()  # [1024, hidden_dim]

    # Select top features by L2 norm
    feature_norms = decoder_weights.norm(dim=0)  # [hidden_dim]
    top_feature_indices = torch.topk(feature_norms, min(num_features, len(feature_norms))).indices

    # Extract top features
    top_features = decoder_weights[:, top_feature_indices]  # [1024, num_features]

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ResNet50 Decoder Feature Weights Analysis (Two-Level Sparsity)', fontsize=14, fontweight='bold')

    # 1. Heatmap of top features
    ax = axes[0, 0]
    im = ax.imshow(top_features.numpy(), cmap='seismic', aspect='auto', vmin=-1, vmax=1)
    ax.set_title(f'Top {num_features} Feature Weights (Channels × Features)')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('ResNet50 Channel (0-1023)')
    plt.colorbar(im, ax=ax)

    # 2. Channel usage per feature
    ax = axes[0, 1]
    channel_usage = (top_features.abs() > 1e-3).sum(dim=0).numpy()
    ax.bar(range(len(channel_usage)), channel_usage, color='green', alpha=0.7)
    ax.set_title('Channel Usage Per Feature')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('# Active Channels')
    ax.grid(True, alpha=0.3)

    # 3. Feature importance per channel
    ax = axes[1, 0]
    channel_importance = top_features.abs().sum(dim=1).numpy()
    ax.bar(range(len(channel_importance)), channel_importance, color='orange', alpha=0.7)
    ax.set_title('Total Feature Importance Per Channel')
    ax.set_xlabel('ResNet50 Channel Index')
    ax.set_ylabel('Cumulative Weight')
    ax.grid(True, alpha=0.3)

    # 4. Weight distribution
    ax = axes[1, 1]
    weights_flat = top_features.flatten().numpy()
    ax.hist(weights_flat, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(weights_flat), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(weights_flat):.3f}')
    ax.set_title('Decoder Weight Distribution')
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Decoder weights visualization saved to {save_path}")
    plt.close()


def compare_input_vs_reconstruction(
    model,
    sample_activations: torch.Tensor,
    num_samples: int = 3,
    save_path: str = 'resnet50_reconstruction_comparison.png'
):
    """
    Side-by-side comparison of input activations vs reconstructions
    across all 1024 channels (averaged).

    Args:
        model: Trained MultiChannelConvSAE model
        sample_activations: [N, 1024, 14, 14] - Sample activation maps
        num_samples: Number of samples to compare
        save_path: Path to save the visualization
    """
    model.eval()
    device = next(model.parameters()).device

    # Select random samples
    indices = np.random.choice(len(sample_activations), size=min(num_samples, len(sample_activations)), replace=False)
    samples = sample_activations[indices].to(device)

    # Forward pass
    with torch.no_grad():
        reconstructions, sparse_features = model(samples, use_topk=True)

    # Move to CPU
    samples = samples.cpu()
    reconstructions = reconstructions.cpu()
    sparse_features = sparse_features.cpu()

    # Create comparison figure
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('ResNet50 SAE: Input vs Reconstruction Comparison (Two-Level Sparsity)', fontsize=14, fontweight='bold')

    for i in range(num_samples):
        input_act = samples[i]  # [1024, 14, 14]
        recon_act = reconstructions[i]  # [1024, 14, 14]

        # Average across channels
        input_avg = input_act.mean(dim=0)  # [14, 14]
        recon_avg = recon_act.mean(dim=0)  # [14, 14]
        diff = torch.abs(input_avg - recon_avg)

        # Compute metrics
        mse = F.mse_loss(recon_act, input_act).item()
        mae = F.l1_loss(recon_act, input_act).item()
        active_channels = (input_act.abs().sum(dim=(1, 2)) > 1e-6).sum().item()

        # Plot input
        axes[i, 0].imshow(input_avg.numpy(), cmap='viridis', aspect='auto')
        axes[i, 0].set_title(f'Sample {i+1}: Input\n({active_channels}/1024 active ch)')
        axes[i, 0].axis('off')

        # Plot reconstruction
        axes[i, 1].imshow(recon_avg.numpy(), cmap='viridis', aspect='auto')
        axes[i, 1].set_title(f'Reconstruction\nMSE: {mse:.4f}')
        axes[i, 1].axis('off')

        # Plot difference
        axes[i, 2].imshow(diff.numpy(), cmap='Reds', aspect='auto')
        axes[i, 2].set_title(f'Absolute Difference\nMAE: {mae:.4f}')
        axes[i, 2].axis('off')

        # Plot sparsity pattern
        sparse_activity = (sparse_features[i] > 0).float().sum(dim=0)  # [14, 14]
        axes[i, 3].imshow(sparse_activity.numpy(), cmap='hot', aspect='auto')
        axes[i, 3].set_title(f'Feature Density\n(# active features/position)')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Reconstruction comparison saved to {save_path}")
    plt.close()
