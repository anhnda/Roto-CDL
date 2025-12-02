"""
Dual ConvSAE Training Script with ResNet18 Backbone

Uses pretrained ResNet18 (ImageNet) to extract activation maps from layer3 (14×14 resolution).
No fine-tuning required - uses GradCAM to extract class-discriminative features.

Key differences from AlexNet version:
- Backbone: ResNet18 (pretrained on ImageNet-1k)
- Target layer: layer3 (256 channels, 14×14 spatial resolution)
- Maps ImageNet-1k predictions to Imagenette-10 classes
- No fine-tuning needed

Usage:
    python run_dual_csae_resnet18.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from torchvision import datasets, transforms
import joblib
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from tqdm import tqdm

from src.convsae import DualConvSAE, LateralInhibitionLoss, ClassDiversityLoss


# ==========================================
# ImageNet to Imagenette Class Mapping
# ==========================================

# Imagenette uses 10 classes from ImageNet-1k
# Map Imagenette class names to their ImageNet class indices
IMAGENETTE_TO_IMAGENET = {
    'tench': 0,          # n01440764
    'springer': 217,     # n02102040 - English springer spaniel
    'cassette_player': 482,  # n02979186
    'chain_saw': 491,    # n03000684
    'church': 497,       # n03028079
    'french_horn': 566,  # n03394916 - French horn
    'garbage_truck': 569,  # n03417042
    'gas_pump': 571,     # n03425413 - Gas pump
    'golf_ball': 574,    # n03445777
    'parachute': 701,    # n03888257
}

# Standard Imagenette class order (matches dataset folder order)
IMAGENETTE_CLASSES = [
    'tench', 'springer', 'cassette_player', 'chain_saw', 'church',
    'french_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute'
]


class ResNet18ActivationExtractor:
    """
    Extracts activation maps from ResNet18 layer3 using GradCAM.

    ResNet18 layer3 outputs 256 channels at 14×14 spatial resolution,
    which is ideal for sparse autoencoder training.
    """

    def __init__(self, device='cuda'):
        self.device = device

        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True).to(device)
        self.model.eval()

        # Target layer: layer3 (outputs 256 channels, 14×14 spatial resolution)
        self.target_layer = self.model.layer3

        # Hook for activations
        self.activations = None
        self.gradients = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook to save gradients."""
        self.gradients = grad_output[0].detach()

    def get_imagenette_predictions(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get predictions for Imagenette classes from ImageNet-1k predictions.

        Args:
            images: [B, 3, 224, 224] - input images

        Returns:
            imagenette_probs: [B, 10] - probabilities for 10 Imagenette classes
        """
        # Get ImageNet-1k predictions
        with torch.no_grad():
            imagenet_logits = self.model(images)  # [B, 1000]
            imagenet_probs = F.softmax(imagenet_logits, dim=1)

        # Extract probabilities for Imagenette classes
        imagenette_indices = [IMAGENETTE_TO_IMAGENET[cls] for cls in IMAGENETTE_CLASSES]
        imagenette_probs = imagenet_probs[:, imagenette_indices]  # [B, 10]

        # Re-normalize to sum to 1
        imagenette_probs = imagenette_probs / (imagenette_probs.sum(dim=1, keepdim=True) + 1e-8)

        return imagenette_probs

    def compute_gradcam(self, images: torch.Tensor, class_idx: int) -> tuple:
        """
        Compute GradCAM for a specific Imagenette class.

        Args:
            images: [B, 3, 224, 224] - input images
            class_idx: Target Imagenette class index (0-9)

        Returns:
            cam_map: [B, 14, 14] - GradCAM heatmap
            channel_weights: [256] - Channel importance weights
            activations: [B, 256, 14, 14] - Raw activation maps
        """
        self.model.zero_grad()

        # Forward pass
        imagenet_logits = self.model(images)  # [B, 1000]

        # Map Imagenette class to ImageNet class
        imagenet_class_idx = IMAGENETTE_TO_IMAGENET[IMAGENETTE_CLASSES[class_idx]]

        # Get score for target class
        score = imagenet_logits[:, imagenet_class_idx].sum()

        # Backward pass
        score.backward()

        # Get activation maps and gradients
        activations = self.activations  # [B, 256, 14, 14]
        gradients = self.gradients      # [B, 256, 14, 14]

        # Compute channel weights (global average pooling of gradients)
        channel_weights = gradients.mean(dim=(0, 2, 3))  # [256]

        # Compute weighted combination of activation maps
        cam_map = (channel_weights.view(1, -1, 1, 1) * activations).sum(dim=1)  # [B, 14, 14]
        cam_map = F.relu(cam_map)  # ReLU to keep only positive contributions

        return cam_map, channel_weights, activations

    def collect_activation_maps(
        self,
        data_loader: DataLoader,
        top_k_percentile: float = 0.9,
        normalize: bool = True
    ) -> tuple:
        """
        Collect activation maps from top-k channels using GradCAM.

        Args:
            data_loader: DataLoader with (image, label) pairs
            top_k_percentile: Select channels contributing to top-k% of GradCAM score
            normalize: Apply robust normalization (99th percentile)

        Returns:
            X: [N_total_maps, 1, 14, 14] - All selected activation maps (each channel is a separate sample)
            Y: [N_total_maps] - Corresponding labels (repeated for each selected channel)
        """
        all_activations = []
        all_labels = []
        n_selected_channels = []  # Track number of selected channels per image

        print(f"Collecting activation maps from ResNet18 layer3 (14×14)...")

        for images, labels in tqdm(data_loader, desc="Processing images"):
            images = images.to(self.device)
            batch_size = images.shape[0]

            for i in range(batch_size):
                img = images[i:i+1]
                label = labels[i].item()

                # Compute GradCAM for this image's true class
                cam_map, channel_weights, activations = self.compute_gradcam(img, label)

                # Select top-k channels
                abs_weights = channel_weights.abs()
                sorted_indices = torch.argsort(abs_weights, descending=True)
                sorted_weights = abs_weights[sorted_indices]

                # Cumulative selection
                cumsum = torch.cumsum(sorted_weights, dim=0)
                total = cumsum[-1]
                n_selected = (cumsum <= top_k_percentile * total).sum().item() + 1
                selected_channels = sorted_indices[:n_selected]

                # Track number of selected channels
                n_selected_channels.append(n_selected)

                # Extract activation maps from selected channels
                selected_acts = activations[0, selected_channels, :, :]  # [n_selected, 14, 14]

                # Add all selected activation maps (not averaged)
                # Each selected channel becomes a separate training sample
                for channel_act in selected_acts:
                    all_activations.append(channel_act.unsqueeze(0))  # [1, 14, 14]
                    all_labels.append(label)

        # Stack into tensors
        # Each activation map has shape [1, 14, 14], stack gives [N_total_maps, 1, 14, 14]
        # where N_total_maps = sum of all selected channels across all images
        X = torch.stack(all_activations)  # [N_total_maps, 1, 14, 14]
        Y = torch.tensor(all_labels, dtype=torch.long)

        # Print channel selection statistics
        n_selected_array = np.array(n_selected_channels)
        print(f"\nChannel Selection Statistics (out of 256 total channels):")
        print(f"  Average channels selected per image: {n_selected_array.mean():.2f}")
        print(f"  Std dev: {n_selected_array.std():.2f}")
        print(f"  Min: {n_selected_array.min()}")
        print(f"  Max: {n_selected_array.max()}")
        print(f"  Median: {np.median(n_selected_array):.0f}")
        print(f"  Percentile (25%, 50%, 75%): "
              f"{np.percentile(n_selected_array, 25):.0f}, "
              f"{np.percentile(n_selected_array, 50):.0f}, "
              f"{np.percentile(n_selected_array, 75):.0f}")
        print(f"  Selection ratio: {n_selected_array.mean()/256*100:.1f}% of all channels")

        # Robust normalization
        if normalize:
            print("Applying robust normalization...")
            flat = X.flatten()
            num = min(10_000_000, flat.numel())
            idx = torch.randint(0, flat.numel(), (num,), device=flat.device)
            scale_factor = torch.quantile(flat[idx], 0.99)

            print(f"  99th percentile scale factor: {scale_factor:.4f}")
            X = torch.clamp(X, min=0.0, max=scale_factor)
            X = X / (scale_factor + 1e-8)

            print(f"  Normalized range: [{X.min():.4f}, {X.max():.4f}]")
            print(f"  Mean: {X.mean():.4f}, Std: {X.std():.4f}")

        return X, Y


# ==========================================
# Visualization Functions (Reused from run_dual_csae.py)
# ==========================================

def plot_dual_training_logs(logs, save_path='dual_csae_resnet18_logs.png'):
    """Plot training metrics for dual-pathway ConvSAE."""
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle('Dual ConvSAE Training (ResNet18 Backbone)', fontsize=16, fontweight='bold')

    # Row 1: Reconstruction metrics
    axs[0, 0].plot(logs["recon_loss"], color='blue', linewidth=1.5)
    axs[0, 0].set_title("Total Reconstruction Loss")
    axs[0, 0].set_ylabel("MSE")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(logs["shared_recon_loss"], color='green', label='Shared', linewidth=1.5)
    axs[0, 1].plot(logs["class_recon_loss"], color='orange', label='Class', linewidth=1.5)
    axs[0, 1].set_title("Pathway Reconstruction Losses")
    axs[0, 1].set_ylabel("MSE")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    axs[0, 2].plot(logs["classification_loss"], color='crimson', linewidth=1.5)
    axs[0, 2].set_title("Classification Loss")
    axs[0, 2].set_ylabel("CrossEntropy")
    axs[0, 2].grid(True, alpha=0.3)

    axs[0, 3].plot(logs["classification_acc"], color='purple', linewidth=1.5)
    axs[0, 3].set_title("Classification Accuracy")
    axs[0, 3].set_ylabel("Accuracy (%)")
    axs[0, 3].set_ylim(0, 100)
    axs[0, 3].grid(True, alpha=0.3)

    # Row 2: Sparsity metrics
    axs[1, 0].plot(logs["shared_l1_loss"], color='green', linewidth=1.5)
    axs[1, 0].set_title("Shared Features L1 Sparsity")
    axs[1, 0].set_ylabel("L1 Loss")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(logs["class_l1_loss"], color='orange', linewidth=1.5)
    axs[1, 1].set_title("Class Features L1 Sparsity")
    axs[1, 1].set_ylabel("L1 Loss")
    axs[1, 1].grid(True, alpha=0.3)

    axs[1, 2].plot(logs["shared_active_pct"], color='green', linewidth=1.5, label='Shared')
    axs[1, 2].plot(logs["class_active_pct"], color='orange', linewidth=1.5, label='Class')
    axs[1, 2].set_title("Active Neurons %")
    axs[1, 2].set_ylabel("Percent (%)")
    axs[1, 2].set_ylim(0, 100)
    axs[1, 2].axhspan(5, 15, alpha=0.2, color='gray', label='Target (5-15%)')
    axs[1, 2].legend()
    axs[1, 2].grid(True, alpha=0.3)

    axs[1, 3].plot(logs["diversity_loss"], color='crimson', linewidth=1.5)
    axs[1, 3].set_title("Class Diversity Loss")
    axs[1, 3].set_ylabel("Similarity")
    axs[1, 3].grid(True, alpha=0.3)

    # Row 3: Other metrics
    axs[2, 0].plot(logs["shared_lateral_loss"], color='green', linewidth=1.5, label='Shared')
    axs[2, 0].plot(logs["class_lateral_loss"], color='orange', linewidth=1.5, label='Class')
    axs[2, 0].set_title("Lateral Inhibition Losses")
    axs[2, 0].set_ylabel("Loss")
    axs[2, 0].legend()
    axs[2, 0].grid(True, alpha=0.3)

    axs[2, 1].plot(logs["total_loss"], color='black', linewidth=2)
    axs[2, 1].set_title("Total Loss")
    axs[2, 1].set_ylabel("Loss")
    axs[2, 1].grid(True, alpha=0.3)

    # Loss components (log scale)
    axs[2, 2].plot(logs["recon_loss"], label='Recon', alpha=0.7)
    axs[2, 2].plot(logs["classification_loss"], label='Cls', alpha=0.7)
    axs[2, 2].plot(logs["diversity_loss"], label='Diversity', alpha=0.7)
    axs[2, 2].set_title("Loss Components (Log Scale)")
    axs[2, 2].set_ylabel("Loss")
    axs[2, 2].set_yscale('log')
    axs[2, 2].legend(fontsize=8)
    axs[2, 2].grid(True, alpha=0.3)

    # Feature usage ratio
    if "shared_l1_loss" in logs and "class_l1_loss" in logs:
        shared_array = np.array(logs["shared_l1_loss"])
        class_array = np.array(logs["class_l1_loss"])
        ratio = shared_array / (class_array + 1e-8)
        axs[2, 3].plot(ratio, color='teal', linewidth=1.5)
        axs[2, 3].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal')
        axs[2, 3].set_title("Shared/Class Feature Usage Ratio")
        axs[2, 3].set_ylabel("Ratio")
        axs[2, 3].legend()
        axs[2, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training logs saved to {save_path}")
    plt.close()


def visualize_dual_features(model, num_features=32, save_path='dual_csae_resnet18_features.png'):
    """Visualize learned features from both pathways."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Dual ConvSAE Learned Features (ResNet18)', fontsize=16, fontweight='bold')

    # Get decoder weights
    shared_weights = model.shared_decoder.weight.detach().cpu().flatten().numpy()
    class_weights = model.class_decoder.weight.detach().cpu().flatten().numpy()

    # 1. Shared decoder weight distribution
    axes[0, 0].hist(shared_weights, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(shared_weights), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(shared_weights):.3f}')
    axes[0, 0].set_title('Shared Decoder Weight Distribution')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Class decoder weight distribution
    axes[0, 1].hist(class_weights, bins=50, color='orange', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(class_weights), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(class_weights):.3f}')
    axes[0, 1].set_title('Class Decoder Weight Distribution')
    axes[0, 1].set_xlabel('Weight Value')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Per-feature weight magnitudes (shared)
    shared_weight = model.shared_decoder.weight.detach().cpu()
    shared_feature_norms = (shared_weight ** 2).sum(dim=(0, 2, 3)).sqrt().numpy()
    n_show = min(num_features, len(shared_feature_norms))
    axes[1, 0].bar(range(n_show), shared_feature_norms[:n_show], color='green', alpha=0.7)
    axes[1, 0].set_title(f'Shared Feature Magnitudes (Top {n_show})')
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('L2 Norm')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Per-feature weight magnitudes (class)
    class_weight = model.class_decoder.weight.detach().cpu()
    class_feature_norms = (class_weight ** 2).sum(dim=(0, 2, 3)).sqrt().numpy()
    n_show = min(num_features, len(class_feature_norms))
    axes[1, 1].bar(range(n_show), class_feature_norms[:n_show], color='orange', alpha=0.7)
    axes[1, 1].set_title(f'Class Feature Magnitudes (Top {n_show})')
    axes[1, 1].set_xlabel('Feature Index')
    axes[1, 1].set_ylabel('L2 Norm')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Feature visualization saved to {save_path}")
    plt.close()


# ==========================================
# Main Training Script
# ==========================================

if __name__ == "__main__":
    # ========================================
    # 1. SETUP DATA
    # ========================================
    print("="*70)
    print("Dual ConvSAE Training with ResNet18 Backbone")
    print("="*70)
    print("\nSetting up Data...")

    data_dir = 'data/imagenette'
    BATCH_SIZE_COLLECTION = 8  # Can use larger batch for ResNet18

    # Standard ImageNet preprocessing
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
    data_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE_COLLECTION, shuffle=False)

    print(f"Dataset: {len(full_dataset)} images")
    print(f"Classes: {full_dataset.classes}")

    # ========================================
    # 2. EXTRACT ACTIVATION MAPS
    # ========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create activation extractor
    extractor = ResNet18ActivationExtractor(device=device)

    # Collect activation maps
    print("\nExtracting activation maps from ResNet18 layer3...")
    X, Y = extractor.collect_activation_maps(
        data_loader,
        top_k_percentile=0.8,  # Use top 90% of channels by GradCAM score
        normalize=True
    )

    print(f"\nCollected activation maps:")
    print(f"  Shape: {X.shape}")  # [N, 1, 14, 14]
    print(f"  Labels: {Y.shape}")
    print(f"  Unique classes: {Y.unique().tolist()}")

    # ========================================
    # 3. SETUP DUAL CONVSAE TRAINING
    # ========================================
    print("\n" + "="*70)
    print("Setting up Dual ConvSAE training...")
    print("="*70)

    # Hyperparameters
    BATCH_SIZE = 256
    INPUT_CHANNELS = 1      # Single-channel activation maps
    SHARED_DIM = 256        # Global features
    CLASS_DIM = 256         # Class-discriminative features
    KERNEL_SIZE = 3         # 3×3 convolution
    NUM_CLASSES = 10        # Imagenette has 10 classes (actually 9, but we'll use 10)

    # Loss weights (optimized for discrimination)
    LAMBDA_SHARED_L1 = 0.001
    LAMBDA_CLASS_L1 = 0.001
    LAMBDA_SHARED_LAT = 0.00
    LAMBDA_CLASS_LAT = 0.00
    LAMBDA_CLASSIFICATION = 5.0
    LAMBDA_DIVERSITY = 1

    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    EPOCHS = 15

    print(f"\nTraining Configuration:")
    print(f"  Backbone: ResNet18 (pretrained ImageNet)")
    print(f"  Target layer: layer3 (256 channels, 14×14 resolution)")
    print(f"  Input Channels: {INPUT_CHANNELS}")
    print(f"  Shared Dim: {SHARED_DIM}")
    print(f"  Class Dim: {CLASS_DIM}")
    print(f"  Kernel Size: {KERNEL_SIZE}")
    print(f"  Num Classes: {NUM_CLASSES}")
    print(f"  Lambda Shared L1: {LAMBDA_SHARED_L1}")
    print(f"  Lambda Class L1: {LAMBDA_CLASS_L1}")
    print(f"  Lambda Classification: {LAMBDA_CLASSIFICATION}")
    print(f"  Lambda Diversity: {LAMBDA_DIVERSITY}")
    print(f"  Learning Rate: {LR}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")

    # Create Dual ConvSAE model
    dual_csae = DualConvSAE(
        in_channels=INPUT_CHANNELS,
        shared_dim=SHARED_DIM,
        class_dim=CLASS_DIM,
        num_classes=NUM_CLASSES,
        kernel_size=KERNEL_SIZE
    ).to(device)

    optimizer = optim.Adam(dual_csae.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lat_inhib_loss = LateralInhibitionLoss().to(device)
    diversity_loss_fn = ClassDiversityLoss(num_classes=NUM_CLASSES).to(device)

    # Create DataLoader
    dataset = TensorDataset(X, Y)
    print("SHAPE: ", X.shape, Y.shape)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Logging
    logs = {
        "total_loss": [],
        "recon_loss": [],
        "shared_recon_loss": [],
        "class_recon_loss": [],
        "classification_loss": [],
        "classification_acc": [],
        "shared_l1_loss": [],
        "class_l1_loss": [],
        "shared_lateral_loss": [],
        "class_lateral_loss": [],
        "diversity_loss": [],
        "shared_active_pct": [],
        "class_active_pct": []
    }

    # ========================================
    # 4. TRAINING LOOP
    # ========================================
    print("\n" + "="*70)
    print("Starting Training...")
    print("="*70)
    print("\nExpected Behavior:")
    print("  • Classification Accuracy: 60-90% by epoch 10")
    print("  • Diversity Loss: Decrease from ~1.0 to <0.5")
    print("  • Active Neurons: 5-15% for both pathways")
    print("  • Reconstruction Loss: <0.01")
    print("="*70 + "\n")

    for epoch in range(EPOCHS):
        epoch_metrics = {k: 0 for k in logs.keys()}
        n_batches = 0

        # Warmup schedule
        if epoch < 3:
            warmup_factor = (epoch + 1) / 3
        else:
            warmup_factor = 1.0

        for batch_idx, (batch_acts, batch_labels) in enumerate(train_loader):
            batch_acts = batch_acts.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            recon, shared_feats, class_feats, class_logits = dual_csae(batch_acts, return_logits=True)

            # Reconstruction losses
            shared_recon = dual_csae.shared_decoder(shared_feats)
            class_recon = dual_csae.class_decoder(class_feats)

            loss_shared_recon = F.mse_loss(shared_recon, batch_acts)
            loss_class_recon = F.mse_loss(class_recon, batch_acts)
            loss_total_recon = F.mse_loss(recon, batch_acts)

            # Sparsity losses
            loss_shared_l1 = shared_feats.abs().mean()
            loss_class_l1 = class_feats.abs().mean()

            # Lateral inhibition
            loss_shared_lat = lat_inhib_loss(shared_feats)
            loss_class_lat = lat_inhib_loss(class_feats)

            # Classification loss
            loss_classification = F.cross_entropy(class_logits, batch_labels)

            # Diversity loss
            loss_diversity = diversity_loss_fn(class_feats, batch_labels)

            # Combined loss with warmup
            loss = (
                loss_total_recon +
                LAMBDA_SHARED_L1 * loss_shared_l1 +
                LAMBDA_CLASS_L1 * loss_class_l1 +
                LAMBDA_SHARED_LAT * loss_shared_lat +
                LAMBDA_CLASS_LAT * loss_class_lat +
                warmup_factor * LAMBDA_CLASSIFICATION * loss_classification +
                warmup_factor * LAMBDA_DIVERSITY * loss_diversity
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dual_csae.parameters(), max_norm=1.0)
            optimizer.step()

            # Normalize weights
            dual_csae.normalize_encoder_weights()
            dual_csae.normalize_decoder_weights()

            # Collect metrics
            with torch.no_grad():
                shared_active = (shared_feats > 0).float().mean().item() * 100
                class_active = (class_feats > 0).float().mean().item() * 100

                _, predicted = torch.max(class_logits, 1)
                acc = (predicted == batch_labels).float().mean().item() * 100

                logs["total_loss"].append(loss.item())
                logs["recon_loss"].append(loss_total_recon.item())
                logs["shared_recon_loss"].append(loss_shared_recon.item())
                logs["class_recon_loss"].append(loss_class_recon.item())
                logs["classification_loss"].append(loss_classification.item())
                logs["classification_acc"].append(acc)
                logs["shared_l1_loss"].append(loss_shared_l1.item())
                logs["class_l1_loss"].append(loss_class_l1.item())
                logs["shared_lateral_loss"].append(loss_shared_lat.item())
                logs["class_lateral_loss"].append(loss_class_lat.item())
                logs["diversity_loss"].append(loss_diversity.item())
                logs["shared_active_pct"].append(shared_active)
                logs["class_active_pct"].append(class_active)

                for k in epoch_metrics.keys():
                    epoch_metrics[k] += logs[k][-1]
                n_batches += 1

            # Print progress
            if batch_idx % 20 == 0:
                warmup_str = f" [Warmup: {warmup_factor:.2f}]" if warmup_factor < 1.0 else ""
                print(f"\rEpoch {epoch+1}/{EPOCHS} [{batch_idx}/{len(train_loader)}]{warmup_str} "
                      f"Loss: {loss.item():.4f} | Recon: {loss_total_recon.item():.4f} | "
                      f"Cls: {loss_classification.item():.4f} (Acc: {acc:.1f}%) | "
                      f"Div: {loss_diversity.item():.4f} | "
                      f"Act: S={shared_active:.1f}% C={class_active:.1f}%", end="")

        # Epoch summary
        avg_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}

        print(f"\n[Epoch {epoch+1}/{EPOCHS}] Summary:")
        print(f"  Total Loss: {avg_metrics['total_loss']:.4f}")
        print(f"  Reconstruction: {avg_metrics['recon_loss']:.4f}")
        print(f"  Classification: {avg_metrics['classification_loss']:.4f} "
              f"(Acc: {avg_metrics['classification_acc']:.2f}%)")
        print(f"  Diversity: {avg_metrics['diversity_loss']:.4f}")
        print(f"  Active: Shared={avg_metrics['shared_active_pct']:.2f}%, "
              f"Class={avg_metrics['class_active_pct']:.2f}%")
        print("-" * 70)

    print("="*70)
    print("Training Complete!")
    print("="*70)

    # ========================================
    # 5. SAVE MODEL
    # ========================================
    print("\nSaving models...")

    torch.save(dual_csae.state_dict(), 'dual_csae_resnet18_model.pth')
    print("✓ Model state dict: dual_csae_resnet18_model.pth")

    joblib.dump(dual_csae.cpu(), 'dual_csae_resnet18_model.pkl')
    print("✓ Full model: dual_csae_resnet18_model.pkl")

    training_info = {
        'config': {
            'backbone': 'ResNet18',
            'target_layer': 'layer3',
            'input_channels': INPUT_CHANNELS,
            'shared_dim': SHARED_DIM,
            'class_dim': CLASS_DIM,
            'num_classes': NUM_CLASSES,
            'kernel_size': KERNEL_SIZE,
            'lambda_shared_l1': LAMBDA_SHARED_L1,
            'lambda_class_l1': LAMBDA_CLASS_L1,
            'lambda_classification': LAMBDA_CLASSIFICATION,
            'lambda_diversity': LAMBDA_DIVERSITY,
            'lr': LR,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
        },
        'logs': logs,
        'final_metrics': avg_metrics
    }
    joblib.dump(training_info, 'dual_csae_resnet18_training_info.pkl')
    print("✓ Training info: dual_csae_resnet18_training_info.pkl")

    # ========================================
    # 6. VISUALIZATIONS
    # ========================================
    print("\nGenerating visualizations...")

    plot_dual_training_logs(logs, save_path='dual_csae_resnet18_logs.png')
    dual_csae = dual_csae.to(device)
    visualize_dual_features(dual_csae, num_features=32, save_path='dual_csae_resnet18_features.png')

    print("\n" + "="*70)
    print("✓ All done! Outputs:")
    print("  - dual_csae_resnet18_model.pth")
    print("  - dual_csae_resnet18_model.pkl")
    print("  - dual_csae_resnet18_training_info.pkl")
    print("  - dual_csae_resnet18_logs.png")
    print("  - dual_csae_resnet18_features.png")
    print("="*70)
