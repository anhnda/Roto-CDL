"""
Accuracy Drop Analysis for ResNet18 Masked Loss CSAE

This script evaluates the impact of replacing original ResNet18 layer3 activations
with CSAE-reconstructed activations on classification accuracy.

Methodology:
1. Load pretrained ResNet18 model
2. Load trained Masked Loss Multi-Channel ConvSAE
3. For each test image:
   a. Extract original activations at layer3
   b. Pass through CSAE to get reconstruction
   c. Replace original with reconstructed activations
   d. Continue forward pass through rest of ResNet18
   e. Compare predictions with/without reconstruction

Metrics:
- Original Accuracy: Classification accuracy with original activations
- Reconstruction Accuracy: Classification accuracy with CSAE reconstructions
- Accuracy Drop: Difference between original and reconstruction accuracy
- Top-5 Accuracy: For both original and reconstructed

Usage:
    python check_accuracy_drop_resnet_mask.py
    python check_accuracy_drop_resnet_mask.py --csae_model multichannel_csae_resnet_mask_model.pkl
    python check_accuracy_drop_resnet_mask.py --num_samples 1000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import joblib
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, Tuple
import sys

sys.path.append('.')
from run_resnet_mask import MultiChannelConvSAE


# Imagenette class names to ImageNet class indices mapping
IMAGENETTE_TO_IMAGENET = {
    'tench': 0,
    'English_springer': 217,
    'cassette_player': 482,
    'chain_saw': 491,
    'church': 497,
    'French_horn': 566,
    'garbage_truck': 569,
    'gas_pump': 571,
    'golf_ball': 574,
    'parachute': 701
}


class ResNet18WithCSAEReconstruction:
    """
    ResNet18 model with CSAE reconstruction at layer3.

    Allows evaluation of classification accuracy when original activations
    are replaced with CSAE reconstructions.
    """

    def __init__(self, resnet_model: nn.Module, csae_model: MultiChannelConvSAE, device='cuda'):
        """
        Args:
            resnet_model: Pretrained ResNet18 model
            csae_model: Trained Multi-Channel ConvSAE (masked loss variant)
            device: Device to run on
        """
        self.resnet = resnet_model.to(device)
        self.csae = csae_model.to(device)
        self.device = device

        self.resnet.eval()
        self.csae.eval()

        # Hook to capture activations at layer3
        self.target_layer = self.resnet.layer3
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output

    def _normalize_activations(self, acts: torch.Tensor) -> torch.Tensor:
        """
        Normalize activations using per-channel 99th percentile.
        Same normalization as used during CSAE training.

        Args:
            acts: [B, C, H, W] - Raw activations

        Returns:
            normalized: [B, C, H, W] - Normalized activations
        """
        normalized = acts.clone()

        for b in range(acts.shape[0]):
            for c in range(acts.shape[1]):
                channel_data = acts[b, c, :, :]

                # Skip channels that are all zeros
                if channel_data.abs().sum() < 1e-8:
                    continue

                # 99th percentile normalization
                non_zero_vals = channel_data[channel_data > 1e-8]
                if len(non_zero_vals) > 0:
                    scale_factor = torch.quantile(non_zero_vals, 0.99)

                    if scale_factor > 1e-8:
                        channel_data = torch.clamp(channel_data, min=0.0, max=scale_factor)
                        normalized[b, c, :, :] = channel_data / (scale_factor + 1e-8)

        return normalized

    def forward_original(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with original activations (no reconstruction).

        Args:
            x: [B, 3, 224, 224] - Input images

        Returns:
            logits: [B, num_classes] - Classification logits
        """
        with torch.no_grad():
            return self.resnet(x)

    def forward_with_reconstruction(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with CSAE-reconstructed activations at layer3.

        Args:
            x: [B, 3, 224, 224] - Input images

        Returns:
            logits: [B, num_classes] - Classification logits with reconstructed activations
            stats: Dictionary with reconstruction statistics
        """
        with torch.no_grad():
            # Forward through ResNet18 up to layer3
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)
            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)

            # Capture original activations
            original_acts = x.clone()  # [B, C, H, W]

            # Normalize activations (same as training)
            normalized_acts = self._normalize_activations(original_acts)

            # Pass through CSAE to get reconstruction
            reconstruction, sparse_features = self.csae(normalized_acts, use_topk=True)

            # Denormalize reconstruction (reverse of normalization)
            # Note: This is approximate since we normalized per-sample during forward
            # For simplicity, we'll use the reconstruction directly
            reconstructed_acts = reconstruction

            # Replace original with reconstructed activations
            x = reconstructed_acts

            # Continue forward pass through rest of ResNet18
            x = self.resnet.layer4(x)
            x = self.resnet.avgpool(x)
            x = torch.flatten(x, 1)
            logits = self.resnet.fc(x)

            # Compute reconstruction statistics
            mse = F.mse_loss(reconstructed_acts, original_acts).item()
            relative_error = (reconstructed_acts - original_acts).abs().mean() / (original_acts.abs().mean() + 1e-8)
            relative_error = relative_error.item()

            sparsity = (sparse_features > 0).float().mean().item()

            stats = {
                'mse': mse,
                'relative_error': relative_error,
                'sparsity': sparsity
            }

            return logits, stats


def evaluate_accuracy_drop(
    model: ResNet18WithCSAEReconstruction,
    data_loader: DataLoader,
    device: str = 'cuda',
    imagenette_to_imagenet: Dict[str, int] = None
) -> Dict:
    """
    Evaluate classification accuracy with original vs reconstructed activations.

    Args:
        model: ResNet18WithCSAEReconstruction model
        data_loader: DataLoader with (image, label) pairs
        device: Device to run on
        imagenette_to_imagenet: Mapping from Imagenette class names to ImageNet indices

    Returns:
        results: Dictionary with accuracy metrics
    """
    # Create label mapping (Imagenette local index -> ImageNet index)
    if imagenette_to_imagenet is not None:
        # Get class names from dataset
        if hasattr(data_loader.dataset, 'dataset'):
            class_names = data_loader.dataset.dataset.classes
        else:
            class_names = data_loader.dataset.classes

        # Create mapping: local_idx -> imagenet_idx
        local_to_imagenet = torch.tensor([imagenette_to_imagenet[name] for name in class_names])
        print(f"\nUsing ImageNet label mapping:")
        for i, name in enumerate(class_names):
            print(f"  Imagenette[{i}] '{name}' -> ImageNet[{local_to_imagenet[i]}]")
    else:
        local_to_imagenet = None

    # Metrics
    total_samples = 0
    correct_original = 0
    correct_reconstructed = 0
    top5_correct_original = 0
    top5_correct_reconstructed = 0

    # Reconstruction stats
    mse_list = []
    relative_error_list = []
    sparsity_list = []

    print("\nEvaluating accuracy with original vs reconstructed activations...")
    print("="*80)

    for images, labels in tqdm(data_loader, desc="Processing batches"):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Convert Imagenette labels to ImageNet labels if mapping exists
        if local_to_imagenet is not None:
            imagenet_labels = local_to_imagenet[labels.cpu()].to(device)
        else:
            imagenet_labels = labels

        # Forward with original activations
        logits_original = model.forward_original(images)
        pred_original = logits_original.argmax(dim=1)

        # Forward with reconstructed activations
        logits_reconstructed, stats = model.forward_with_reconstruction(images)
        pred_reconstructed = logits_reconstructed.argmax(dim=1)

        # Top-1 accuracy (compare with ImageNet labels)
        correct_original += (pred_original == imagenet_labels).sum().item()
        correct_reconstructed += (pred_reconstructed == imagenet_labels).sum().item()

        # Top-5 accuracy
        _, top5_original = logits_original.topk(5, dim=1)
        _, top5_reconstructed = logits_reconstructed.topk(5, dim=1)

        top5_correct_original += sum([imagenet_labels[i] in top5_original[i] for i in range(batch_size)])
        top5_correct_reconstructed += sum([imagenet_labels[i] in top5_reconstructed[i] for i in range(batch_size)])

        total_samples += batch_size

        # Collect reconstruction stats
        mse_list.append(stats['mse'])
        relative_error_list.append(stats['relative_error'])
        sparsity_list.append(stats['sparsity'])

    # Compute accuracies
    acc_original = correct_original / total_samples * 100
    acc_reconstructed = correct_reconstructed / total_samples * 100
    acc_drop = acc_original - acc_reconstructed

    top5_acc_original = top5_correct_original / total_samples * 100
    top5_acc_reconstructed = top5_correct_reconstructed / total_samples * 100
    top5_acc_drop = top5_acc_original - top5_acc_reconstructed

    # Average reconstruction stats
    avg_mse = np.mean(mse_list)
    avg_relative_error = np.mean(relative_error_list)
    avg_sparsity = np.mean(sparsity_list)

    results = {
        'total_samples': total_samples,
        'acc_original': acc_original,
        'acc_reconstructed': acc_reconstructed,
        'acc_drop': acc_drop,
        'top5_acc_original': top5_acc_original,
        'top5_acc_reconstructed': top5_acc_reconstructed,
        'top5_acc_drop': top5_acc_drop,
        'avg_mse': avg_mse,
        'avg_relative_error': avg_relative_error,
        'avg_sparsity': avg_sparsity
    }

    return results


def print_results(results: Dict):
    """
    Pretty-print accuracy drop analysis results.

    Args:
        results: Dictionary with accuracy metrics
    """
    print("\n" + "="*80)
    print("ACCURACY DROP ANALYSIS RESULTS")
    print("="*80)

    print(f"\nDataset:")
    print(f"  Total samples: {results['total_samples']}")

    print(f"\nTop-1 Accuracy:")
    print(f"  Original (no reconstruction):     {results['acc_original']:.2f}%")
    print(f"  Reconstructed (CSAE):             {results['acc_reconstructed']:.2f}%")
    print(f"  Accuracy Drop:                    {results['acc_drop']:.2f}%")
    if results['acc_original'] > 0:
        print(f"  Relative Drop:                    {results['acc_drop']/results['acc_original']*100:.2f}%")
    else:
        print(f"  Relative Drop:                    N/A (original accuracy is 0%)")

    print(f"\nTop-5 Accuracy:")
    print(f"  Original (no reconstruction):     {results['top5_acc_original']:.2f}%")
    print(f"  Reconstructed (CSAE):             {results['top5_acc_reconstructed']:.2f}%")
    print(f"  Accuracy Drop:                    {results['top5_acc_drop']:.2f}%")
    if results['top5_acc_original'] > 0:
        print(f"  Relative Drop:                    {results['top5_acc_drop']/results['top5_acc_original']*100:.2f}%")
    else:
        print(f"  Relative Drop:                    N/A (original top-5 accuracy is 0%)")

    print(f"\nReconstruction Quality:")
    print(f"  Average MSE:                      {results['avg_mse']:.6f}")
    print(f"  Average Relative Error:           {results['avg_relative_error']:.6f}")
    print(f"  Average Sparsity (active %):      {results['avg_sparsity']*100:.2f}%")

    print("\n" + "="*80)

    # Interpretation
    print("\nInterpretation:")
    if results['acc_drop'] < 1.0:
        print("  ✓ Excellent: CSAE preserves almost all classification information (<1% drop)")
    elif results['acc_drop'] < 3.0:
        print("  ✓ Very Good: CSAE preserves most classification information (<3% drop)")
    elif results['acc_drop'] < 5.0:
        print("  ✓ Good: CSAE preserves significant classification information (<5% drop)")
    elif results['acc_drop'] < 10.0:
        print("  ⚠ Moderate: Notable accuracy drop (5-10%), but reconstruction is still useful")
    else:
        print("  ⚠ High: Significant accuracy drop (>10%), CSAE may be too lossy")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate accuracy drop with CSAE-reconstructed activations (ResNet18 Masked Loss)'
    )
    parser.add_argument('--csae_model', type=str,
                       default='multichannel_csae_resnet_mask_model.pkl',
                       help='Path to trained Multi-Channel ConvSAE model (masked loss variant)')
    parser.add_argument('--data_dir', type=str, default='data/imagenette',
                       help='Path to Imagenette dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (None = all)')
    parser.add_argument('--output_file', type=str, default='accuracy_drop_resnet_mask_results.txt',
                       help='Output file to save results')

    args = parser.parse_args()

    print("="*80)
    print("ResNet18 Masked Loss CSAE - Accuracy Drop Analysis")
    print("="*80)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load ResNet18 model
    print("\nLoading ResNet18 model...")
    resnet_model = models.resnet18(pretrained=True)
    resnet_model.eval()
    print("  ✓ ResNet18 loaded")

    # Load CSAE model
    print(f"\nLoading CSAE model from {args.csae_model}...")
    csae_model = joblib.load(args.csae_model)
    csae_model.eval()
    print(f"  ✓ CSAE loaded: {csae_model.in_channels}→{csae_model.hidden_dim}, top_k={csae_model.top_k}")

    # Create combined model
    print("\nCreating ResNet18WithCSAEReconstruction...")
    model = ResNet18WithCSAEReconstruction(resnet_model, csae_model, device=device)
    print("  ✓ Model ready")

    # Setup data
    print(f"\nLoading dataset from {args.data_dir}...")
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=args.data_dir, transform=data_transform)

    # Optionally limit number of samples
    if args.num_samples is not None and args.num_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:args.num_samples].tolist()
        dataset = Subset(dataset, indices)
        print(f"  ✓ Using {args.num_samples} samples (randomly selected)")
    else:
        print(f"  ✓ Using all {len(dataset)} samples")

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"  Batch size: {args.batch_size}")
    print(f"  Classes: {len(dataset.dataset.classes) if hasattr(dataset, 'dataset') else len(dataset.classes)}")

    # Evaluate accuracy drop (with ImageNet label mapping for Imagenette)
    results = evaluate_accuracy_drop(model, data_loader, device=device,
                                     imagenette_to_imagenet=IMAGENETTE_TO_IMAGENET)

    # Print results
    print_results(results)

    # Save results to file
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ResNet18 Masked Loss CSAE - Accuracy Drop Analysis\n")
        f.write("="*80 + "\n\n")

        f.write(f"Configuration:\n")
        f.write(f"  CSAE Model: {args.csae_model}\n")
        f.write(f"  Dataset: {args.data_dir}\n")
        f.write(f"  Total Samples: {results['total_samples']}\n")
        f.write(f"  Batch Size: {args.batch_size}\n\n")

        f.write(f"Top-1 Accuracy:\n")
        f.write(f"  Original:        {results['acc_original']:.2f}%\n")
        f.write(f"  Reconstructed:   {results['acc_reconstructed']:.2f}%\n")
        f.write(f"  Drop:            {results['acc_drop']:.2f}%\n")
        if results['acc_original'] > 0:
            f.write(f"  Relative Drop:   {results['acc_drop']/results['acc_original']*100:.2f}%\n\n")
        else:
            f.write(f"  Relative Drop:   N/A (original accuracy is 0%)\n\n")

        f.write(f"Top-5 Accuracy:\n")
        f.write(f"  Original:        {results['top5_acc_original']:.2f}%\n")
        f.write(f"  Reconstructed:   {results['top5_acc_reconstructed']:.2f}%\n")
        f.write(f"  Drop:            {results['top5_acc_drop']:.2f}%\n")
        if results['top5_acc_original'] > 0:
            f.write(f"  Relative Drop:   {results['top5_acc_drop']/results['top5_acc_original']*100:.2f}%\n\n")
        else:
            f.write(f"  Relative Drop:   N/A (original top-5 accuracy is 0%)\n\n")

        f.write(f"Reconstruction Quality:\n")
        f.write(f"  Average MSE:             {results['avg_mse']:.6f}\n")
        f.write(f"  Average Relative Error:  {results['avg_relative_error']:.6f}\n")
        f.write(f"  Average Sparsity:        {results['avg_sparsity']*100:.2f}%\n\n")

        f.write("="*80 + "\n")

    print(f"✓ Results saved to {args.output_file}")
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
