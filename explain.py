
"""
Explainability Module for Roto-CDL

This module provides tools to explain model predictions by:
1. Using GradCAM to find top influential activation maps (80% cumulative score)
2. Decomposing each activation map using learned dictionary atoms
3. Tracing back activated patterns to input image regions using deconvolution

Usage:
    explainer = ExplanationPipeline(model, phi_learned, device)
    results = explainer.explain_prediction(image, class_idx=None)
    explainer.visualize_explanation(results, save_path='explanation.png')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from typing import Dict, List, Tuple, Optional
import joblib

try:
    from scipy.ndimage import zoom
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Receptive field visualization may be limited.")

from src.gradcam import GradCAM
from src.batch_cdl_large import solve_z_prox_adam, rotated_atoms, reconstruct


class ExplanationPipeline:
    """
    Complete explanation pipeline for Roto-CDL model predictions.

    Args:
        model: Trained classification model (e.g., FineTunedModel)
        phi_learned: Learned dictionary atoms [n_atoms, 1, d, d]
        device: Computing device (cuda/cpu)
        target_layer: Layer to extract activations from (default: layer 5)
        n_rotations: Number of rotations for atoms (default: 4)
    """

    def __init__(
        self,
        model: nn.Module,
        phi_learned: torch.Tensor,
        device: torch.device,
        target_layer: Optional[nn.Module] = None,
        n_rotations: int = 4
    ):
        self.model = model.to(device)
        self.model.eval()  # Set to evaluation mode
        self.phi_learned = phi_learned.to(device)
        self.device = device
        self.n_rotations = n_rotations

        # Default to layer 5 of feature extractor
        if target_layer is None:
            self.target_layer = model.feature_extractor[5]
        else:
            self.target_layer = target_layer

        # Initialize GradCAM
        self.gradcam = GradCAM(model, self.target_layer)

        # Store intermediate activations for deconv
        # Note: Only forward hooks are registered to avoid conflicts with inplace ReLU
        self.layer_activations = {}
        self.layer_gradients = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks for activation storage"""
        def make_forward_hook(name):
            def hook(module, input, output):
                # Clone to avoid in-place modification issues
                self.layer_activations[name] = output.detach().clone()
            return hook

        # Register forward hooks only (backward hooks conflict with inplace ReLU)
        for idx, layer in enumerate(self.model.feature_extractor):
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(make_forward_hook(f'conv_{idx}'))

    def get_top_k_cumulative(
        self,
        weights: torch.Tensor,
        cumulative_threshold: float = 0.8
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Get top-k channel indices based on cumulative score threshold.

        Args:
            weights: Channel weights from GradCAM [C]
            cumulative_threshold: Threshold for cumulative score (e.g., 0.8 for 80%)

        Returns:
            top_indices: List of selected channel indices
            cumulative_scores: Cumulative scores for all channels
        """
        # Sort weights in descending order
        sorted_weights, sorted_indices = torch.sort(weights, descending=True)

        # Compute cumulative sum
        total_weight = sorted_weights.sum()
        cumulative_sum = torch.cumsum(sorted_weights, dim=0)
        cumulative_scores = cumulative_sum / total_weight

        # Find cutoff index where cumulative score exceeds threshold
        cutoff_idx = torch.where(cumulative_scores >= cumulative_threshold)[0]

        if len(cutoff_idx) > 0:
            k = cutoff_idx[0].item() + 1  # +1 because we want to include this index
        else:
            k = len(weights)  # Use all channels if threshold not reached

        top_indices = sorted_indices[:k].cpu().tolist()

        return top_indices, cumulative_scores

    def solve_sparse_code_for_map(
        self,
        activation_map: torch.Tensor,
        lr_z: float = 3.0,
        sparsity: float = 0.005,
        n_steps: int = 50
    ) -> torch.Tensor:
        """
        Solve for sparse code Z given an activation map and learned dictionary.

        Args:
            activation_map: Single activation map [H, W]
            lr_z: Learning rate for Z optimization
            sparsity: Sparsity penalty coefficient
            n_steps: Number of optimization steps

        Returns:
            Z: Sparse code [1, n_atoms, n_rotations, H, W]
        """
        # Add batch and channel dimensions: [H, W] -> [1, 1, H, W]
        X = activation_map.unsqueeze(0).unsqueeze(0).to(self.device)

        # Solve for Z using solve_z_prox_adam
        Z = solve_z_prox_adam(
            X,
            self.phi_learned,
            self.n_rotations,
            lr_z,
            sparsity,
            n_steps
        )

        return Z  # [1, n_atoms, n_rotations, H, W]

    def deconv_trace_position(
        self,
        activation_map: torch.Tensor,
        position: Tuple[int, int],
        class_idx: int,
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        Trace back a specific position in activation map to input image using guided backprop.

        Args:
            activation_map: Activation map [H, W]
            position: (h, w) position in activation map
            class_idx: Target class index
            image: Input image [1, 3, H, W]

        Returns:
            receptive_field_mask: Gradient-based attribution map [H_img, W_img]
        """
        self.model.eval()
        image_input = image.clone().detach().requires_grad_(True)

        # Forward pass
        with torch.enable_grad():
            logits = self.model(image_input)

            # Get activation at target layer
            target_activation = self.layer_activations.get('conv_5', self.gradcam.activations)

            if target_activation is None:
                raise ValueError("Target activation not found")

            # Create mask for specific position
            # target_activation shape: [1, C, H, W]
            h, w = position
            activation_value = target_activation[0, :, h, w].sum()

            # Backward pass
            self.model.zero_grad()
            activation_value.backward(retain_graph=True)

            # Get gradient w.r.t. input
            if image_input.grad is not None:
                gradient = image_input.grad.detach()

                # Compute saliency as absolute gradient magnitude
                saliency = gradient.abs().sum(dim=1).squeeze()  # [H_img, W_img]

                # Normalize
                saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            else:
                saliency = torch.zeros(image.shape[2:], device=self.device)

        return saliency

    def get_receptive_field_mask(
        self,
        position: Tuple[int, int],
        activation_shape: Tuple[int, int],
        input_shape: Tuple[int, int],
        stride: int = 16,
        kernel_size: int = 11
    ) -> np.ndarray:
        """
        Compute approximate receptive field mask for a position in activation map.

        This is a geometric approximation based on network architecture.
        For AlexNet layer 5: effective stride ≈ 16, effective kernel ≈ 51x51

        Args:
            position: (h, w) in activation map
            activation_shape: (H_act, W_act)
            input_shape: (H_img, W_img)
            stride: Effective stride from input to this layer
            kernel_size: Effective receptive field size

        Returns:
            mask: Binary mask [H_img, W_img]
        """
        h_act, w_act = position
        H_img, W_img = input_shape

        # Compute center position in input image
        center_h = int(h_act * stride + stride / 2)
        center_w = int(w_act * stride + stride / 2)

        # Create Gaussian mask centered at receptive field
        yy, xx = np.ogrid[:H_img, :W_img]

        # Effective receptive field for layer 5 in AlexNet:
        # Conv1: 11x11, stride 4
        # Pool1: 3x3, stride 2
        # Conv2: 5x5, stride 1
        # Pool2: 3x3, stride 2
        # Effective stride = 4 * 2 * 1 * 2 = 16
        # Effective RF ≈ 51x51

        sigma = 25  # Roughly half of receptive field size
        gaussian = np.exp(-((xx - center_w)**2 + (yy - center_h)**2) / (2 * sigma**2))

        # Normalize
        mask = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min() + 1e-8)

        return mask

    def explain_prediction(
        self,
        image: torch.Tensor,
        class_idx: Optional[int] = None,
        cumulative_threshold: float = 0.8,
        lr_z: float = 3.0,
        sparsity: float = 0.005,
        n_steps_z: int = 50,
        top_atoms_per_map: int = 5
    ) -> Dict:
        """
        Complete explanation pipeline for a single image.

        Args:
            image: Input image [1, 3, H, W]
            class_idx: Target class (None = predicted class)
            cumulative_threshold: Cumulative score threshold for activation maps
            lr_z: Learning rate for sparse coding
            sparsity: Sparsity penalty
            n_steps_z: Optimization steps for Z
            top_atoms_per_map: Number of top atoms to analyze per map

        Returns:
            results: Dictionary containing:
                - 'input_image': Original input
                - 'predicted_class': Predicted/target class
                - 'gradcam_weights': All channel weights
                - 'top_indices': Selected channel indices
                - 'cumulative_scores': Cumulative scores
                - 'activation_maps': List of top activation maps
                - 'sparse_codes': List of Z for each map
                - 'top_atoms_info': List of dicts with atom info per map
                - 'receptive_fields': List of receptive field masks
        """
        image = image.to(self.device)
        self.model.eval()

        # Step 1: Get GradCAM weights
        print("Step 1/5: Computing GradCAM weights...")
        weights, cam, predicted_class = self.gradcam.forward(image, class_idx=class_idx)

        # Step 2: Select top-k activation maps (80% cumulative)
        print(f"Step 2/5: Selecting top activation maps (cumulative threshold: {cumulative_threshold})...")
        top_indices, cumulative_scores = self.get_top_k_cumulative(weights, cumulative_threshold)

        print(f"  → Selected {len(top_indices)} / {len(weights)} channels")
        print(f"  → Cumulative score: {cumulative_scores[len(top_indices)-1]:.2%}")

        # Step 3: Get activation maps for selected channels
        print("Step 3/5: Extracting activation maps...")
        activations = self.gradcam.activations.squeeze()  # [C, H, W]
        activation_maps = [activations[idx].cpu() for idx in top_indices]

        # Step 4: Solve sparse codes for each activation map
        print("Step 4/5: Solving sparse codes with learned dictionary...")
        sparse_codes = []
        top_atoms_info = []

        for idx, act_map in enumerate(activation_maps):
            Z = self.solve_sparse_code_for_map(act_map, lr_z, sparsity, n_steps_z)
            sparse_codes.append(Z)

            # Analyze top atoms for this map
            # Z shape: [1, n_atoms, n_rotations, H, W]
            Z_magnitude = Z.squeeze(0).sum(dim=1)  # Sum over rotations: [n_atoms, H, W]
            atom_importance = Z_magnitude.sum(dim=(1, 2))  # [n_atoms]

            # Get top atoms
            top_atom_vals, top_atom_indices = torch.topk(atom_importance, min(top_atoms_per_map, len(atom_importance)))

            atoms_info = {
                'channel_idx': top_indices[idx],
                'gradcam_weight': weights[top_indices[idx]].item(),
                'top_atom_indices': top_atom_indices.cpu().tolist(),
                'top_atom_scores': top_atom_vals.cpu().tolist(),
                'atom_activation_maps': [Z_magnitude[i].cpu() for i in top_atom_indices],
                'full_Z': Z.cpu()
            }
            top_atoms_info.append(atoms_info)

            print(f"  → Map {idx+1}/{len(activation_maps)}: Channel {top_indices[idx]}, "
                  f"Top atoms: {top_atom_indices[:3].tolist()}")

        # Step 5: Compute receptive fields for top activation positions
        print("Step 5/5: Computing receptive field masks...")
        receptive_fields = []

        H_act, W_act = activation_maps[0].shape
        H_img, W_img = image.shape[2:]

        for idx, (act_map, atoms_info) in enumerate(zip(activation_maps, top_atoms_info)):
            # Find position of maximum activation
            max_pos = torch.argmax(act_map.flatten()).item()
            h_max = max_pos // W_act
            w_max = max_pos % W_act

            # Compute receptive field mask
            rf_mask = self.get_receptive_field_mask(
                (h_max, w_max),
                (H_act, W_act),
                (H_img, W_img)
            )

            receptive_fields.append({
                'channel_idx': atoms_info['channel_idx'],
                'max_position': (h_max, w_max),
                'mask': rf_mask
            })

        print("\n✓ Explanation complete!")

        return {
            'input_image': image.cpu(),
            'predicted_class': predicted_class,
            'gradcam_weights': weights.cpu(),
            'top_indices': top_indices,
            'cumulative_scores': cumulative_scores.cpu(),
            'activation_maps': activation_maps,
            'sparse_codes': sparse_codes,
            'top_atoms_info': top_atoms_info,
            'receptive_fields': receptive_fields,
            'gradcam_heatmap': cam.cpu()
        }

    def visualize_explanation(
        self,
        results: Dict,
        save_path: Optional[str] = None,
        max_maps_to_show: int = 3,
        max_atoms_per_map: int = 4,
        figsize: Tuple[int, int] = (20, 12)
    ):
        """
        Visualize complete explanation results.

        Layout:
        - Row 1: Original image, GradCAM heatmap, overlay
        - Row 2+: For each top activation map:
            - Activation map
            - Top-4 atoms from dictionary
            - Receptive field on input

        Args:
            results: Results from explain_prediction()
            save_path: Path to save figure
            max_maps_to_show: Maximum number of activation maps to visualize
            max_atoms_per_map: Maximum atoms to show per map
            figsize: Figure size
        """
        n_maps = min(max_maps_to_show, len(results['activation_maps']))

        # Create figure with GridSpec
        n_rows = 1 + n_maps  # 1 header row + n_maps detail rows
        n_cols = 6  # Image, GradCAM, overlay + atom1, atom2, atom3, RF

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

        # Denormalize input image
        input_img = results['input_image'].squeeze(0).numpy()  # [3, H, W]
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        input_denorm = input_img * std + mean
        input_denorm = np.clip(input_denorm, 0, 1)
        input_denorm = np.transpose(input_denorm, (1, 2, 0))  # [H, W, 3]

        # Row 0: Overview
        ax_img = fig.add_subplot(gs[0, 0:2])
        ax_img.imshow(input_denorm)
        ax_img.set_title(f'Input Image\nPredicted Class: {results["predicted_class"]}',
                        fontweight='bold', fontsize=12)
        ax_img.axis('off')

        ax_gradcam = fig.add_subplot(gs[0, 2:4])
        gradcam_heatmap = results['gradcam_heatmap'].numpy()
        ax_gradcam.imshow(gradcam_heatmap, cmap='jet')
        ax_gradcam.set_title(f'GradCAM Heatmap\n(Layer 5)', fontweight='bold', fontsize=12)
        ax_gradcam.axis('off')

        ax_overlay = fig.add_subplot(gs[0, 4:6])
        # Resize heatmap to match input
        H_img, W_img = input_denorm.shape[:2]
        H_heat, W_heat = gradcam_heatmap.shape

        if SCIPY_AVAILABLE:
            zoom_factors = (H_img / H_heat, W_img / W_heat)
            heatmap_resized = zoom(gradcam_heatmap, zoom_factors, order=1)
        else:
            # Fallback using PIL
            heatmap_pil = Image.fromarray((gradcam_heatmap * 255).astype(np.uint8))
            heatmap_resized = heatmap_pil.resize((W_img, H_img), Image.BILINEAR)
            heatmap_resized = np.array(heatmap_resized) / 255.0

        ax_overlay.imshow(input_denorm)
        ax_overlay.imshow(heatmap_resized, cmap='jet', alpha=0.4)
        ax_overlay.set_title(f'Overlay\n{len(results["top_indices"])} channels selected (80%)',
                            fontweight='bold', fontsize=12)
        ax_overlay.axis('off')

        # Rows 1+: Per-map details
        for map_idx in range(n_maps):
            row = 1 + map_idx

            atoms_info = results['top_atoms_info'][map_idx]
            act_map = results['activation_maps'][map_idx].numpy()
            rf_info = results['receptive_fields'][map_idx]

            # Column 0: Activation map
            ax_act = fig.add_subplot(gs[row, 0])
            im = ax_act.imshow(act_map, cmap='viridis')
            ax_act.set_title(f'Channel {atoms_info["channel_idx"]}\n'
                           f'Weight: {atoms_info["gradcam_weight"]:.4f}',
                           fontsize=10, fontweight='bold')
            ax_act.axis('off')
            plt.colorbar(im, ax=ax_act, fraction=0.046)

            # Mark max position
            h_max, w_max = rf_info['max_position']
            ax_act.plot(w_max, h_max, 'r*', markersize=15)

            # Columns 1-4: Top atoms
            n_atoms_show = min(max_atoms_per_map, len(atoms_info['top_atom_indices']))
            for atom_i in range(n_atoms_show):
                col = 1 + atom_i
                ax_atom = fig.add_subplot(gs[row, col])

                atom_idx = atoms_info['top_atom_indices'][atom_i]
                atom_score = atoms_info['top_atom_scores'][atom_i]
                atom_img = self.phi_learned[atom_idx, 0].cpu().numpy()

                ax_atom.imshow(atom_img, cmap='gray')
                ax_atom.set_title(f'Atom {atom_idx}\nScore: {atom_score:.3f}',
                                fontsize=9)
                ax_atom.axis('off')

            # Column 5: Receptive field on input
            ax_rf = fig.add_subplot(gs[row, 5])
            ax_rf.imshow(input_denorm)
            ax_rf.imshow(rf_info['mask'], cmap='Reds', alpha=0.5)
            ax_rf.set_title(f'Receptive Field\nPos: {rf_info["max_position"]}',
                          fontsize=10, fontweight='bold')
            ax_rf.axis('off')

        plt.suptitle('Roto-CDL Explanation Pipeline', fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Saved visualization to: {save_path}")

        plt.tight_layout()
        plt.show()

    def visualize_atom_activation_detail(
        self,
        results: Dict,
        map_idx: int = 0,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Detailed visualization of atom activations for a specific activation map.

        Shows spatial activation patterns for each top atom.

        Args:
            results: Results from explain_prediction()
            map_idx: Which activation map to visualize (0-indexed)
            save_path: Path to save figure
            figsize: Figure size
        """
        atoms_info = results['top_atoms_info'][map_idx]
        act_map = results['activation_maps'][map_idx].numpy()

        n_atoms = len(atoms_info['top_atom_indices'])
        n_cols = 3
        n_rows = int(np.ceil((n_atoms + 1) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        # First subplot: original activation map
        axes[0].imshow(act_map, cmap='viridis')
        axes[0].set_title(f'Channel {atoms_info["channel_idx"]} Activation\n'
                         f'GradCAM: {atoms_info["gradcam_weight"]:.4f}',
                         fontweight='bold')
        axes[0].axis('off')

        # Remaining subplots: atom-specific activations
        for i, (atom_idx, atom_score) in enumerate(zip(
            atoms_info['top_atom_indices'],
            atoms_info['top_atom_scores']
        )):
            ax = axes[i + 1]

            # Show atom activation map (summed over rotations)
            atom_act_map = atoms_info['atom_activation_maps'][i].numpy()

            im = ax.imshow(atom_act_map, cmap='hot')
            ax.set_title(f'Atom {atom_idx} Activation\nScore: {atom_score:.3f}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Hide unused subplots
        for i in range(n_atoms + 1, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'Detailed Atom Activations - Map {map_idx}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved to: {save_path}")

        plt.show()


def load_learned_dictionary(dict_path: str, device: torch.device) -> torch.Tensor:
    """
    Load learned dictionary from file.

    Args:
        dict_path: Path to .pkl file with learned atoms
        device: Device to load to

    Returns:
        phi_learned: Dictionary tensor [n_atoms, 1, d, d]
    """
    phi_learned = joblib.load(dict_path)
    if isinstance(phi_learned, np.ndarray):
        phi_learned = torch.from_numpy(phi_learned).float()
    return phi_learned.to(device)


if __name__ == "__main__":
    """
    Example usage of the explanation pipeline.
    """
    from torchvision import datasets, transforms
    from src.model import FineTunedModel

    print("="*70)
    print("Roto-CDL Explanation Pipeline - Example")
    print("="*70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    print("\n1. Loading model...")
    model = FineTunedModel(num_classes=10).to(device)
    model.load_state_dict(torch.load('weights/finetune_weights.pth', map_location=device))
    model.eval()
    print("   ✓ Model loaded")

    # Load learned dictionary
    print("\n2. Loading learned dictionary...")
    phi_learned = load_learned_dictionary('phi_learned_imagenette.pkl', device)
    print(f"   ✓ Dictionary loaded: {phi_learned.shape}")

    # Load test image
    print("\n3. Loading test image...")
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root='data/imagenette', transform=data_transform)
    test_image, true_label = dataset[0]  # Get first image
    test_image = test_image.unsqueeze(0)  # Add batch dimension
    print(f"   ✓ Image loaded, true label: {true_label}")

    # Create explainer
    print("\n4. Creating explanation pipeline...")
    explainer = ExplanationPipeline(
        model=model,
        phi_learned=phi_learned,
        device=device,
        target_layer=model.feature_extractor[5],  # Layer 5
        n_rotations=4
    )
    print("   ✓ Pipeline ready")

    # Generate explanation
    print("\n5. Generating explanation...\n")
    results = explainer.explain_prediction(
        image=test_image,
        class_idx=None,  # Use predicted class
        cumulative_threshold=0.8,
        lr_z=3.0,
        sparsity=0.005,
        n_steps_z=50,
        top_atoms_per_map=5
    )

    # Visualize
    print("\n6. Visualizing results...")
    explainer.visualize_explanation(
        results,
        save_path='explanation_output.png',
        max_maps_to_show=3,
        max_atoms_per_map=4
    )

    # Detailed atom visualization for first map
    print("\n7. Generating detailed atom activation visualization...")
    explainer.visualize_atom_activation_detail(
        results,
        map_idx=0,
        save_path='atom_activations_detail.png'
    )

    print("\n" + "="*70)
    print("✓ Complete! Check 'explanation_output.png' and 'atom_activations_detail.png'")
    print("="*70)
