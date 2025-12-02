"""
Dual ConvSAE Training Script

Trains a dual-pathway ConvSAE that learns:
1. Shared features: Global patterns common across all classes
2. Class-specific features: Discriminative patterns for classification

Usage:
    python run_dual_csae.py
"""

from src.activation import ActivationMapCollector
from src.model import FineTunedModel
from src.convsae import DualConvSAE, LateralInhibitionLoss, ClassDiversityLoss
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm


def plot_dual_training_logs(logs, save_path='dual_csae_training_logs.png'):
    """
    Plots training metrics for dual-pathway ConvSAE.
    """
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle('Dual ConvSAE Training Diagnostics', fontsize=16, fontweight='bold')

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
    axs[0, 2].set_title("Classification Loss (Class Pathway)")
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
    axs[1, 3].set_title("Class Diversity Loss (Lower = More Discriminative)")
    axs[1, 3].set_ylabel("Similarity")
    axs[1, 3].grid(True, alpha=0.3)

    # Row 3: Other losses and total
    axs[2, 0].plot(logs["shared_lateral_loss"], color='green', linewidth=1.5, label='Shared')
    axs[2, 0].plot(logs["class_lateral_loss"], color='orange', linewidth=1.5, label='Class')
    axs[2, 0].set_title("Lateral Inhibition Losses")
    axs[2, 0].set_ylabel("Loss")
    axs[2, 0].legend()
    axs[2, 0].grid(True, alpha=0.3)

    axs[2, 1].plot(logs["total_loss"], color='black', linewidth=2)
    axs[2, 1].set_title("Total Loss (All Components)")
    axs[2, 1].set_ylabel("Loss")
    axs[2, 1].grid(True, alpha=0.3)

    # Loss components breakdown (log scale)
    axs[2, 2].plot(logs["recon_loss"], label='Recon', alpha=0.7)
    axs[2, 2].plot(logs["classification_loss"], label='Cls', alpha=0.7)
    axs[2, 2].plot(logs["shared_l1_loss"], label='Shared L1', alpha=0.7)
    axs[2, 2].plot(logs["class_l1_loss"], label='Class L1', alpha=0.7)
    axs[2, 2].plot(logs["diversity_loss"], label='Diversity', alpha=0.7)
    axs[2, 2].set_title("Loss Components (Log Scale)")
    axs[2, 2].set_ylabel("Loss")
    axs[2, 2].set_yscale('log')
    axs[2, 2].legend(fontsize=8)
    axs[2, 2].grid(True, alpha=0.3)

    # Feature usage ratio
    # Calculate ratio of shared vs class feature importance
    if "shared_l1_loss" in logs and "class_l1_loss" in logs:
        shared_array = np.array(logs["shared_l1_loss"])
        class_array = np.array(logs["class_l1_loss"])
        # Avoid division by zero
        ratio = shared_array / (class_array + 1e-8)
        axs[2, 3].plot(ratio, color='teal', linewidth=1.5)
        axs[2, 3].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal (ratio=1)')
        axs[2, 3].set_title("Shared/Class Feature Usage Ratio")
        axs[2, 3].set_ylabel("Ratio (Shared L1 / Class L1)")
        axs[2, 3].legend()
        axs[2, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training logs saved to {save_path}")
    plt.close()


def visualize_dual_features(model, num_features=32, save_path='dual_csae_features.png'):
    """
    Visualize learned features from both pathways.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Dual ConvSAE Learned Features', fontsize=16, fontweight='bold')

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
    shared_feature_norms = model.shared_decoder.weight.detach().cpu().norm(dim=(0, 2, 3)).numpy()
    n_show = min(num_features, len(shared_feature_norms))
    axes[1, 0].bar(range(n_show), shared_feature_norms[:n_show], color='green', alpha=0.7)
    axes[1, 0].set_title(f'Shared Feature Magnitudes (Top {n_show})')
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('L2 Norm')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Per-feature weight magnitudes (class)
    class_feature_norms = model.class_decoder.weight.detach().cpu().norm(dim=(0, 2, 3)).numpy()
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


if __name__ == "__main__":
    # ========================================
    # 1. SETUP DATA
    # ========================================
    print("Setting up Data...")
    data_dir = 'data/imagenette'
    BATCH_SIZE_COLLECTION = 1  # Keep 1 for collection to save VRAM

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
    data_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE_COLLECTION)

    # ========================================
    # 2. COLLECT FEATURE MAPS
    # ========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = FineTunedModel(num_classes=10).to(device)
    model.load_state_dict(torch.load('weights/finetune_weights.pth'))
    target_layer = model.feature_extractor[5]

    collector = ActivationMapCollector(model, target_layer, device=device)

    # Collect data with class labels
    print("Collecting Activation Maps...")
    X, Y = collector.collect_maps(data_loader, output_tensor=True, device=device, return_labels=True)
    print(f"Collected {len(X)} activation maps with shape {X.shape}")
    print(f"Labels shape: {Y.shape}, unique classes: {Y.unique().tolist()}")

    # ========================================
    # 3. ROBUST NORMALIZATION
    # ========================================
    print("Applying robust normalization...")
    flat = X.flatten()
    num = min(10_000_000, flat.numel())
    idx = torch.randint(0, flat.numel(), (num,), device=flat.device)
    scale_factor = torch.quantile(flat[idx], 0.99)

    print(f"Robust Scale Factor: {scale_factor:.4f}")
    X = torch.clamp(X, min=0.0, max=scale_factor)
    X = X / scale_factor

    # Verify data range
    print(f"Data range after normalization: [{X.min():.4f}, {X.max():.4f}]")
    print(f"Data mean: {X.mean():.4f}, std: {X.std():.4f}")

    # ========================================
    # 4. SETUP DUAL CONVSAE TRAINING
    # ========================================
    print("\nSetting up Dual ConvSAE training...")

    # Hyperparameters
    BATCH_SIZE = 256
    INPUT_CHANNELS = X.shape[1]  # Should be 1
    SHARED_DIM = 256  # Global features
    CLASS_DIM = 256   # Class-discriminative features
    KERNEL_SIZE = 3   # 3x3 convolution
    NUM_CLASSES = 10  # Imagenette

    # Loss weights
    LAMBDA_SHARED_L1 = 0.001    # Sparsity for shared features (reduced to allow more features)
    LAMBDA_CLASS_L1 = 0.001     # Sparsity for class features (much lower - prioritize discrimination)
    LAMBDA_SHARED_LAT = 0.00   # Lateral inhibition for shared
    LAMBDA_CLASS_LAT = 0.00    # Lateral inhibition for class (reduced)
    LAMBDA_CLASSIFICATION = 10.0 # Classification loss weight (INCREASED - make it priority)
    LAMBDA_DIVERSITY = 2      # Diversity loss weight (INCREASED - force class separation)

    LR = 3e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 15

    print(f"Training Configuration:")
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
    # 5. TRAINING LOOP
    # ========================================
    print("\nStarting Dual ConvSAE Training...")
    print("=" * 70)
    print("\nExpected Behavior & Tuning Guide:")
    print("  • Classification Accuracy: Should reach 60-90% by epoch 10")
    print("    - If stuck at ~10-20%: INCREASE lambda_classification (try 10.0)")
    print("    - If >95%: Model might be overfitting, DECREASE lambda_class_l1")
    print("  • Diversity Loss: Should decrease from ~1.0 to <0.5")
    print("    - If stays >0.8: Classes using same features, INCREASE lambda_diversity")
    print("    - If <0.2: Too much separation, may need more shared features")
    print("  • Active Neurons: Target 5-15% for both pathways")
    print("    - If <2%: Too sparse, DECREASE lambda_l1")
    print("    - If >30%: Not sparse enough, INCREASE lambda_l1")
    print("  • Reconstruction Loss: Should stay <0.01 (data normalized to [0,1])")
    print("=" * 70 + "\n")

    for epoch in range(EPOCHS):
        epoch_metrics = {k: 0 for k in logs.keys()}
        n_batches = 0

        # Warmup schedule for classification and diversity losses
        # Start with lower weights and gradually increase to full strength
        if epoch < 3:
            warmup_factor = (epoch + 1) / 3  # 0.33, 0.67, 1.0
        else:
            warmup_factor = 1.0

        for batch_idx, (batch_acts, batch_labels) in enumerate(train_loader):
            batch_acts = batch_acts.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            recon, shared_feats, class_feats, class_logits = dual_csae(batch_acts, return_logits=True)

            # ===== RECONSTRUCTION LOSSES =====
            # Individual pathway reconstructions
            shared_recon = dual_csae.shared_decoder(shared_feats)
            class_recon = dual_csae.class_decoder(class_feats)

            loss_shared_recon = F.mse_loss(shared_recon, batch_acts)
            loss_class_recon = F.mse_loss(class_recon, batch_acts)
            loss_total_recon = F.mse_loss(recon, batch_acts)

            # ===== SPARSITY LOSSES =====
            loss_shared_l1 = shared_feats.abs().mean()
            loss_class_l1 = class_feats.abs().mean()

            # ===== LATERAL INHIBITION =====
            loss_shared_lat = lat_inhib_loss(shared_feats)
            loss_class_lat = lat_inhib_loss(class_feats)

            # ===== CLASSIFICATION LOSS =====
            loss_classification = F.cross_entropy(class_logits, batch_labels)

            # ===== DIVERSITY LOSS =====
            loss_diversity = diversity_loss_fn(class_feats, batch_labels)

            # ===== COMBINED LOSS =====
            # Apply warmup to classification and diversity losses
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
                # Active neurons
                shared_active = (shared_feats > 0).float().mean().item() * 100
                class_active = (class_feats > 0).float().mean().item() * 100

                # Classification accuracy
                _, predicted = torch.max(class_logits, 1)
                acc = (predicted == batch_labels).float().mean().item() * 100

                # Log all metrics
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

                # Accumulate for epoch summary
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
        print(f"  Reconstruction: {avg_metrics['recon_loss']:.4f} "
              f"(Shared: {avg_metrics['shared_recon_loss']:.4f}, Class: {avg_metrics['class_recon_loss']:.4f})")
        print(f"  Classification: {avg_metrics['classification_loss']:.4f} "
              f"(Acc: {avg_metrics['classification_acc']:.2f}%)")
        print(f"  Diversity: {avg_metrics['diversity_loss']:.4f}")
        print(f"  Active Neurons: Shared={avg_metrics['shared_active_pct']:.2f}%, "
              f"Class={avg_metrics['class_active_pct']:.2f}%")
        print("-" * 70)

    print("=" * 70)
    print("Training Complete!")

    # ========================================
    # 6. SAVE MODEL
    # ========================================
    print("\nSaving trained Dual ConvSAE model...")

    torch.save(dual_csae.state_dict(), 'dual_csae_model.pth')
    print("✓ Model state dict saved to: dual_csae_model.pth")

    joblib.dump(dual_csae.cpu(), 'dual_csae_model.pkl')
    print("✓ Full model saved to: dual_csae_model.pkl")

    training_info = {
        'config': {
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
    joblib.dump(training_info, 'dual_csae_training_info.pkl')
    print("✓ Training info saved to: dual_csae_training_info.pkl")

    # ========================================
    # 7. VISUALIZATIONS
    # ========================================
    print("\nGenerating visualizations...")

    plot_dual_training_logs(logs, save_path='dual_csae_training_logs.png')
    dual_csae = dual_csae.to(device)
    visualize_dual_features(dual_csae, num_features=32, save_path='dual_csae_features.png')

    print("\n" + "=" * 70)
    print("✓ All done! Outputs:")
    print("  - dual_csae_model.pth: Model state dict")
    print("  - dual_csae_model.pkl: Full model (joblib)")
    print("  - dual_csae_training_info.pkl: Training config and logs")
    print("  - dual_csae_training_logs.png: Training metrics visualization")
    print("  - dual_csae_features.png: Learned features visualization")
    print("=" * 70)
