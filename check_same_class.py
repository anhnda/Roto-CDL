"""
Same-Class Atom Activation Analysis

This script analyzes whether images from the same class share similar activated atoms.
It helps validate that the learned dictionary has captured semantically meaningful patterns.

Usage:
    python check_same_class.py --class_name tench --num_images 10
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse
import os

from explain import ExplanationPipeline, load_learned_dictionary
from src.model import FineTunedModel


class SameClassAtomAnalyzer:
    """
    Analyzes atom activation patterns across images from the same class.
    """

    def __init__(
        self,
        explainer: ExplanationPipeline,
        class_names: List[str],
        dataset: datasets.ImageFolder
    ):
        self.explainer = explainer
        self.class_names = class_names
        self.dataset = dataset
        self.results_by_class = {}

    def analyze_class(
        self,
        class_name: str,
        num_images: int = 10,
        cumulative_threshold: float = 0.8,
        **explain_kwargs
    ) -> Dict:
        """
        Analyze atom activations for multiple images from the same class.

        Args:
            class_name: Name of the class to analyze
            num_images: Number of images to sample from the class
            cumulative_threshold: Threshold for selecting activation maps
            **explain_kwargs: Additional arguments for explain_prediction

        Returns:
            analysis_results: Dictionary containing:
                - 'class_name': Class name
                - 'class_idx': Class index
                - 'image_results': List of explanation results for each image
                - 'atom_frequency': Frequency of each atom across all images
                - 'atom_importance': Average importance score for each atom
                - 'top_shared_atoms': Atoms that appear in most images
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

        # Run explanation pipeline on each image
        image_results = []
        atom_frequency = defaultdict(int)  # Count how many images activate each atom
        atom_importance = defaultdict(list)  # Track importance scores for each atom

        for idx, img_idx in enumerate(sampled_indices):
            image, label = self.dataset[img_idx]
            image = image.unsqueeze(0)  # Add batch dimension

            print(f"\n[{idx+1}/{num_samples}] Processing image {img_idx}...")

            # Run explanation
            result = self.explainer.explain_prediction(
                image=image,
                cumulative_threshold=cumulative_threshold,
                **explain_kwargs
            )

            # Extract atom information
            activated_atoms = set()
            for atoms_info in result['top_atoms_info']:
                for atom_idx, atom_score in zip(
                    atoms_info['top_atom_indices'],
                    atoms_info['top_atom_scores']
                ):
                    activated_atoms.add(atom_idx)
                    atom_frequency[atom_idx] += 1
                    atom_importance[atom_idx].append(atom_score)

            # Store results
            image_results.append({
                'image_idx': img_idx,
                'explanation': result,
                'activated_atoms': activated_atoms
            })

            print(f"   → Activated atoms: {sorted(list(activated_atoms))[:10]}...")

        # Compute statistics
        n_atoms_total = self.explainer.phi_learned.shape[0]

        # Average importance for each atom
        avg_importance = {
            atom_idx: np.mean(scores)
            for atom_idx, scores in atom_importance.items()
        }

        # Find atoms that appear in most images
        atom_freq_sorted = sorted(atom_frequency.items(), key=lambda x: x[1], reverse=True)

        # Get top shared atoms (appear in at least 50% of images)
        threshold_freq = num_samples * 0.5
        top_shared_atoms = [
            (atom_idx, freq, avg_importance[atom_idx])
            for atom_idx, freq in atom_freq_sorted
            if freq >= threshold_freq
        ]

        print(f"\n{'='*70}")
        print(f"Analysis Summary for '{class_name}':")
        print(f"{'='*70}")
        print(f"Total images analyzed: {num_samples}")
        print(f"Total unique atoms activated: {len(atom_frequency)}/{n_atoms_total}")
        print(f"Atoms appearing in ≥50% of images: {len(top_shared_atoms)}")
        print(f"\nTop 10 most frequently activated atoms:")
        for i, (atom_idx, freq, importance) in enumerate(top_shared_atoms[:10]):
            print(f"  {i+1}. Atom {atom_idx}: {freq}/{num_samples} images ({freq/num_samples*100:.1f}%), "
                  f"avg importance: {importance:.3f}")

        results = {
            'class_name': class_name,
            'class_idx': class_idx,
            'num_images': num_samples,
            'image_results': image_results,
            'atom_frequency': dict(atom_frequency),
            'atom_importance': avg_importance,
            'top_shared_atoms': top_shared_atoms,
            'atom_freq_sorted': atom_freq_sorted
        }

        self.results_by_class[class_name] = results
        return results

    def compare_classes(
        self,
        class_names: List[str],
        num_images_per_class: int = 10
    ) -> Dict:
        """
        Compare atom activation patterns across multiple classes.

        Args:
            class_names: List of class names to compare
            num_images_per_class: Number of images to analyze per class

        Returns:
            comparison_results: Dictionary with cross-class analysis
        """
        print(f"\n{'='*70}")
        print(f"Cross-Class Atom Activation Comparison")
        print(f"{'='*70}\n")

        # Analyze each class if not already done
        for class_name in class_names:
            if class_name not in self.results_by_class:
                self.analyze_class(class_name, num_images_per_class)

        # Build atom-class matrix
        n_atoms = self.explainer.phi_learned.shape[0]
        atom_class_matrix = np.zeros((n_atoms, len(class_names)))

        for col_idx, class_name in enumerate(class_names):
            results = self.results_by_class[class_name]
            for atom_idx, freq in results['atom_frequency'].items():
                atom_class_matrix[atom_idx, col_idx] = freq / results['num_images']

        # Find class-specific atoms (high activation in one class, low in others)
        class_specific_atoms = {}
        for col_idx, class_name in enumerate(class_names):
            class_activations = atom_class_matrix[:, col_idx]
            other_activations = atom_class_matrix[:, [i for i in range(len(class_names)) if i != col_idx]].max(axis=1)

            # Atoms that activate strongly in this class but weakly in others
            specificity = class_activations - other_activations
            top_specific_idx = np.argsort(specificity)[-10:][::-1]  # Top 10

            class_specific_atoms[class_name] = [
                (int(idx), float(class_activations[idx]), float(specificity[idx]))
                for idx in top_specific_idx
                if specificity[idx] > 0.2  # At least 20% more frequent
            ]

        print(f"\nClass-Specific Atoms:")
        for class_name, atoms in class_specific_atoms.items():
            print(f"\n{class_name}:")
            if atoms:
                for atom_idx, freq, spec in atoms[:5]:
                    print(f"  Atom {atom_idx}: {freq*100:.1f}% activation, specificity: {spec:.3f}")
            else:
                print("  No highly specific atoms found")

        return {
            'class_names': class_names,
            'atom_class_matrix': atom_class_matrix,
            'class_specific_atoms': class_specific_atoms
        }

    def visualize_class_analysis(
        self,
        class_name: str,
        save_path: str = None,
        top_k_atoms: int = 10,
        show_images: int = 5
    ):
        """
        Visualize atom activation patterns for a single class.

        Creates a figure showing:
        - Sample images from the class
        - Most frequently activated atoms
        - Activation heatmap across images

        Args:
            class_name: Class to visualize
            save_path: Path to save figure
            top_k_atoms: Number of top atoms to show
            show_images: Number of sample images to display
        """
        if class_name not in self.results_by_class:
            raise ValueError(f"No analysis results for class '{class_name}'. Run analyze_class first.")

        results = self.results_by_class[class_name]

        # Setup figure
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1.5, 1.5], hspace=0.3, wspace=0.3)

        # ===== Row 0: Sample Images =====
        ax_samples = fig.add_subplot(gs[0, :])
        ax_samples.axis('off')
        ax_samples.set_title(f"Sample Images from '{class_name}' ({results['num_images']} total analyzed)",
                           fontsize=14, fontweight='bold')

        num_show = min(show_images, len(results['image_results']))
        sample_images = []

        for i in range(num_show):
            img_result = results['image_results'][i]
            img_tensor = img_result['explanation']['input_image'].squeeze(0).numpy()

            # Denormalize
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_denorm = img_tensor * std + mean
            img_denorm = np.clip(img_denorm, 0, 1)
            img_denorm = np.transpose(img_denorm, (1, 2, 0))

            # Create subplot for this image
            ax_img = fig.add_axes([0.05 + i * 0.18, 0.70, 0.15, 0.25])
            ax_img.imshow(img_denorm)
            ax_img.set_title(f"Image {i+1}", fontsize=10)
            ax_img.axis('off')

        # ===== Row 1: Top Atoms =====
        ax_atoms = fig.add_subplot(gs[1, :])
        ax_atoms.set_title(f"Top {top_k_atoms} Most Frequently Activated Atoms",
                          fontsize=14, fontweight='bold')

        top_atoms = results['top_shared_atoms'][:top_k_atoms]
        phi_cpu = self.explainer.phi_learned.cpu()

        for i, (atom_idx, freq, importance) in enumerate(top_atoms):
            ax = fig.add_axes([0.05 + (i % 10) * 0.09, 0.40 - (i // 10) * 0.15, 0.08, 0.12])
            atom_img = phi_cpu[atom_idx, 0].numpy()
            ax.imshow(atom_img, cmap='gray')
            ax.set_title(f"Atom {atom_idx}\n{freq}/{results['num_images']} ({freq/results['num_images']*100:.0f}%)",
                        fontsize=8)
            ax.axis('off')

        ax_atoms.axis('off')

        # ===== Row 2: Activation Heatmap =====
        ax_heatmap = fig.add_subplot(gs[2, 0])

        # Build heatmap matrix: atoms × images
        n_images = len(results['image_results'])
        top_atom_indices = [atom_idx for atom_idx, _, _ in top_atoms]

        heatmap_data = np.zeros((len(top_atom_indices), n_images))

        for img_idx, img_result in enumerate(results['image_results']):
            for row_idx, atom_idx in enumerate(top_atom_indices):
                # Check if this atom was activated in this image
                if atom_idx in img_result['activated_atoms']:
                    # Find the importance score
                    for atoms_info in img_result['explanation']['top_atoms_info']:
                        if atom_idx in atoms_info['top_atom_indices']:
                            score_idx = atoms_info['top_atom_indices'].index(atom_idx)
                            heatmap_data[row_idx, img_idx] = atoms_info['top_atom_scores'][score_idx]
                            break

        sns.heatmap(heatmap_data,
                   xticklabels=[f"Img {i+1}" for i in range(n_images)],
                   yticklabels=[f"Atom {idx}" for idx in top_atom_indices],
                   cmap='YlOrRd',
                   ax=ax_heatmap,
                   cbar_kws={'label': 'Activation Score'})
        ax_heatmap.set_title("Atom Activation Scores Across Images", fontsize=12, fontweight='bold')
        ax_heatmap.set_xlabel("Image", fontsize=10)
        ax_heatmap.set_ylabel("Atom", fontsize=10)

        # ===== Row 2, Col 1: Frequency Bar Chart =====
        ax_bar = fig.add_subplot(gs[2, 1])

        atom_ids = [atom_idx for atom_idx, _, _ in top_atoms]
        frequencies = [freq / results['num_images'] * 100 for _, freq, _ in top_atoms]

        bars = ax_bar.barh(range(len(atom_ids)), frequencies, color='steelblue')
        ax_bar.set_yticks(range(len(atom_ids)))
        ax_bar.set_yticklabels([f"Atom {idx}" for idx in atom_ids])
        ax_bar.set_xlabel("Activation Frequency (%)", fontsize=10)
        ax_bar.set_title("Atom Activation Frequency", fontsize=12, fontweight='bold')
        ax_bar.invert_yaxis()
        ax_bar.grid(axis='x', alpha=0.3)

        # Add percentage labels
        for i, (bar, freq) in enumerate(zip(bars, frequencies)):
            ax_bar.text(freq + 1, i, f'{freq:.1f}%', va='center', fontsize=9)

        plt.suptitle(f"Same-Class Atom Activation Analysis: '{class_name}'",
                    fontsize=16, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Saved visualization to: {save_path}")

        plt.tight_layout()
        plt.show()

    def visualize_atom_activations_on_image(
        self,
        class_name: str,
        image_idx: int = 0,
        top_k_atoms: int = 6,
        save_path: str = None
    ):
        """
        Visualize where specific atoms activate on the input image.

        For each top atom, shows:
        - The atom pattern
        - The activation map (where the atom fires)
        - The input image with activation overlay (showing activated regions)

        Args:
            class_name: Class to visualize
            image_idx: Which image from the analyzed set (0-indexed)
            top_k_atoms: Number of top atoms to show
            save_path: Path to save figure
        """
        if class_name not in self.results_by_class:
            raise ValueError(f"No results for class '{class_name}'. Run analyze_class first.")

        results = self.results_by_class[class_name]

        if image_idx >= len(results['image_results']):
            raise ValueError(f"Image index {image_idx} out of range. Max: {len(results['image_results'])-1}")

        img_result = results['image_results'][image_idx]
        explanation = img_result['explanation']

        # Get input image
        img_tensor = explanation['input_image'].squeeze(0).numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_denorm = img_tensor * std + mean
        img_denorm = np.clip(img_denorm, 0, 1)
        img_denorm = np.transpose(img_denorm, (1, 2, 0))  # [H, W, 3]

        # Get top atoms for this specific image
        all_atom_scores = []
        for atoms_info in explanation['top_atoms_info']:
            for atom_idx, atom_score in zip(
                atoms_info['top_atom_indices'],
                atoms_info['top_atom_scores']
            ):
                all_atom_scores.append((atom_idx, atom_score, atoms_info))

        # Sort by score and get top-k unique atoms
        all_atom_scores.sort(key=lambda x: x[1], reverse=True)
        seen_atoms = set()
        top_atoms_data = []
        for atom_idx, score, atoms_info in all_atom_scores:
            if atom_idx not in seen_atoms and len(top_atoms_data) < top_k_atoms:
                seen_atoms.add(atom_idx)
                # Find the activation map for this atom
                atom_pos = atoms_info['top_atom_indices'].index(atom_idx)
                atom_act_map = atoms_info['atom_activation_maps'][atom_pos].numpy()
                top_atoms_data.append((atom_idx, score, atom_act_map, atoms_info['channel_idx']))

        # Create figure
        n_atoms = len(top_atoms_data)
        fig = plt.figure(figsize=(18, 3 * n_atoms))
        gs = GridSpec(n_atoms, 4, figure=fig, wspace=0.3, hspace=0.4)

        phi_cpu = self.explainer.phi_learned.cpu()

        for row, (atom_idx, score, atom_act_map, channel_idx) in enumerate(top_atoms_data):
            # Column 0: Atom pattern
            ax_atom = fig.add_subplot(gs[row, 0])
            atom_pattern = phi_cpu[atom_idx, 0].numpy()
            ax_atom.imshow(atom_pattern, cmap='gray')
            ax_atom.set_title(f'Atom {atom_idx}\nScore: {score:.3f}', fontsize=11, fontweight='bold')
            ax_atom.axis('off')

            # Column 1: Activation map
            ax_act = fig.add_subplot(gs[row, 1])
            im = ax_act.imshow(atom_act_map, cmap='hot')
            ax_act.set_title(f'Activation Map\n(Channel {channel_idx})', fontsize=11, fontweight='bold')
            ax_act.axis('off')
            plt.colorbar(im, ax=ax_act, fraction=0.046)

            # Column 2: Input image with activation overlay
            ax_overlay = fig.add_subplot(gs[row, 2])

            # Upsample activation map to input image size
            from PIL import Image as PILImage
            H_img, W_img = img_denorm.shape[:2]
            H_act, W_act = atom_act_map.shape

            # Normalize activation map
            act_normalized = (atom_act_map - atom_act_map.min()) / (atom_act_map.max() - atom_act_map.min() + 1e-8)

            # Upsample using PIL
            act_pil = PILImage.fromarray((act_normalized * 255).astype(np.uint8))
            act_upsampled = act_pil.resize((W_img, H_img), PILImage.BILINEAR)
            act_upsampled = np.array(act_upsampled) / 255.0

            # Overlay on input
            ax_overlay.imshow(img_denorm)
            ax_overlay.imshow(act_upsampled, cmap='hot', alpha=0.5)
            ax_overlay.set_title('Activated Regions\n(Overlay)', fontsize=11, fontweight='bold')
            ax_overlay.axis('off')

            # Column 3: Masked input (show only activated regions)
            ax_masked = fig.add_subplot(gs[row, 3])

            # Create mask (threshold at 50% of max activation)
            mask = act_upsampled > 0.5

            # Apply mask to image
            masked_img = img_denorm.copy()
            for c in range(3):
                masked_img[:, :, c] = masked_img[:, :, c] * mask

            # Blend with original for context
            blend_alpha = 0.3
            masked_img = blend_alpha * img_denorm + (1 - blend_alpha) * masked_img

            ax_masked.imshow(masked_img)
            ax_masked.set_title('Highlighted Regions\n(Masked)', fontsize=11, fontweight='bold')
            ax_masked.axis('off')

        plt.suptitle(f"Atom Activations on Input Image - Class: '{class_name}' (Image {image_idx+1})",
                    fontsize=14, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Saved atom activation visualization to: {save_path}")

        plt.tight_layout()
        plt.show()

    def visualize_cross_class_comparison(
        self,
        class_name: str,
        top_k_atoms: int = 6,
        save_path: str = None
    ):
        """
        For a given class, visualize how top atoms activate across different sample images.

        Shows a grid where:
        - Rows: Top atoms
        - Columns: Different sample images from the class
        - Each cell: Activation overlay on input image

        Args:
            class_name: Class to visualize
            top_k_atoms: Number of top atoms to show
            save_path: Path to save figure
        """
        if class_name not in self.results_by_class:
            raise ValueError(f"No results for class '{class_name}'. Run analyze_class first.")

        results = self.results_by_class[class_name]
        top_atoms = results['top_shared_atoms'][:top_k_atoms]

        # Get images to show (up to 5)
        num_images_show = min(5, len(results['image_results']))

        # Create figure
        fig = plt.figure(figsize=(4 * num_images_show, 3 * top_k_atoms))
        gs = GridSpec(top_k_atoms + 1, num_images_show, figure=fig, wspace=0.1, hspace=0.3)

        phi_cpu = self.explainer.phi_learned.cpu()

        # Row 0: Show sample images
        for col, img_result in enumerate(results['image_results'][:num_images_show]):
            img_tensor = img_result['explanation']['input_image'].squeeze(0).numpy()
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_denorm = img_tensor * std + mean
            img_denorm = np.clip(img_denorm, 0, 1)
            img_denorm = np.transpose(img_denorm, (1, 2, 0))

            ax = fig.add_subplot(gs[0, col])
            ax.imshow(img_denorm)
            ax.set_title(f'Image {col+1}', fontsize=10, fontweight='bold')
            ax.axis('off')

        # Rows 1+: Show atom activations
        for row, (atom_idx, freq, importance) in enumerate(top_atoms):
            for col, img_result in enumerate(results['image_results'][:num_images_show]):
                ax = fig.add_subplot(gs[row + 1, col])

                # Get input image
                img_tensor = img_result['explanation']['input_image'].squeeze(0).numpy()
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                img_denorm = img_tensor * std + mean
                img_denorm = np.clip(img_denorm, 0, 1)
                img_denorm = np.transpose(img_denorm, (1, 2, 0))

                # Find activation map for this atom in this image
                atom_act_map = None
                for atoms_info in img_result['explanation']['top_atoms_info']:
                    if atom_idx in atoms_info['top_atom_indices']:
                        atom_pos = atoms_info['top_atom_indices'].index(atom_idx)
                        atom_act_map = atoms_info['atom_activation_maps'][atom_pos].numpy()
                        break

                if atom_act_map is not None:
                    # Upsample and overlay
                    from PIL import Image as PILImage
                    H_img, W_img = img_denorm.shape[:2]

                    act_normalized = (atom_act_map - atom_act_map.min()) / (atom_act_map.max() - atom_act_map.min() + 1e-8)
                    act_pil = PILImage.fromarray((act_normalized * 255).astype(np.uint8))
                    act_upsampled = act_pil.resize((W_img, H_img), PILImage.BILINEAR)
                    act_upsampled = np.array(act_upsampled) / 255.0

                    ax.imshow(img_denorm)
                    ax.imshow(act_upsampled, cmap='hot', alpha=0.6)
                else:
                    # Atom not activated in this image
                    ax.imshow(img_denorm, alpha=0.3)
                    ax.text(0.5, 0.5, 'Not\nActivated', transform=ax.transAxes,
                           ha='center', va='center', fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                if col == 0:
                    ax.set_ylabel(f'Atom {atom_idx}', fontsize=10, fontweight='bold')

                ax.axis('off')

        plt.suptitle(f"Atom Activations Across Images - Class: '{class_name}'",
                    fontsize=14, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Saved cross-image activation visualization to: {save_path}")

        plt.tight_layout()
        plt.show()

    def visualize_multi_class_comparison(
        self,
        class_names: List[str],
        save_path: str = None,
        top_k_atoms: int = 20
    ):
        """
        Visualize atom activation patterns across multiple classes.

        Args:
            class_names: List of classes to compare
            save_path: Path to save figure
            top_k_atoms: Number of atoms to show
        """
        # Get comparison results
        comparison = self.compare_classes(class_names)
        atom_class_matrix = comparison['atom_class_matrix']

        # Find top atoms (most activated across all classes)
        total_activations = atom_class_matrix.sum(axis=1)
        top_atom_indices = np.argsort(total_activations)[-top_k_atoms:][::-1]

        # Filter matrix to top atoms
        heatmap_data = atom_class_matrix[top_atom_indices, :] * 100  # Convert to percentage

        # Create figure
        fig, (ax_heatmap, ax_atoms) = plt.subplots(1, 2, figsize=(18, 10),
                                                   gridspec_kw={'width_ratios': [2, 1]})

        # Heatmap
        sns.heatmap(heatmap_data,
                   xticklabels=class_names,
                   yticklabels=[f"Atom {idx}" for idx in top_atom_indices],
                   cmap='YlOrRd',
                   ax=ax_heatmap,
                   cbar_kws={'label': 'Activation Frequency (%)'},
                   annot=True,
                   fmt='.1f')
        ax_heatmap.set_title("Atom Activation Frequency Across Classes", fontsize=14, fontweight='bold')
        ax_heatmap.set_xlabel("Class", fontsize=12)
        ax_heatmap.set_ylabel("Atom", fontsize=12)

        # Show example atoms
        ax_atoms.axis('off')
        ax_atoms.set_title("Sample Dictionary Atoms", fontsize=14, fontweight='bold')

        phi_cpu = self.explainer.phi_learned.cpu()
        n_show = min(8, len(top_atom_indices))

        for i in range(n_show):
            atom_idx = top_atom_indices[i]
            ax = fig.add_axes([0.68 + (i % 4) * 0.075, 0.55 - (i // 4) * 0.25, 0.06, 0.2])
            atom_img = phi_cpu[atom_idx, 0].numpy()
            ax.imshow(atom_img, cmap='gray')
            ax.set_title(f"Atom {atom_idx}", fontsize=9)
            ax.axis('off')

        plt.suptitle("Cross-Class Atom Activation Comparison",
                    fontsize=16, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Saved cross-class comparison to: {save_path}")

        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze atom activation patterns for same-class images")
    parser.add_argument('--class_name', type=str, default='tench',
                       help='Class name to analyze (default: tench)')
    parser.add_argument('--num_images', type=int, default=10,
                       help='Number of images to analyze per class (default: 10)')
    parser.add_argument('--compare_classes', nargs='+', default=None,
                       help='List of classes to compare (e.g., --compare_classes tench church)')
    parser.add_argument('--output_dir', type=str, default='same_class_analysis',
                       help='Output directory for visualizations')
    parser.add_argument('--cumulative_threshold', type=float, default=0.8,
                       help='Cumulative threshold for activation map selection (default: 0.8)')

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

    # Load dictionary
    print("\nLoading learned dictionary...")
    phi_learned = load_learned_dictionary('phi_learned_imagenette.pkl', device)
    print(f"✓ Dictionary loaded: {phi_learned.shape}")

    # Load dataset
    print("\nLoading dataset...")
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root='data/imagenette', transform=data_transform)
    class_names = dataset.classes
    print(f"✓ Dataset loaded: {len(dataset)} images, {len(class_names)} classes")
    print(f"   Classes: {class_names}")

    # Create explainer
    print("\nCreating explanation pipeline...")
    explainer = ExplanationPipeline(
        model=model,
        phi_learned=phi_learned,
        device=device,
        target_layer=model.feature_extractor[5],
        n_rotations=4
    )
    print("✓ Pipeline ready")

    # Create analyzer
    analyzer = SameClassAtomAnalyzer(explainer, class_names, dataset)

    # Single class analysis
    if not args.compare_classes:
        print(f"\n{'='*70}")
        print(f"Single Class Analysis Mode")
        print(f"{'='*70}")

        results = analyzer.analyze_class(
            class_name=args.class_name,
            num_images=args.num_images,
            cumulative_threshold=args.cumulative_threshold,
            lr_z=3.0,
            sparsity=0.005,
            n_steps_z=50
        )

        # Visualize main analysis
        save_path = os.path.join(args.output_dir, f'{args.class_name}_analysis.png')
        analyzer.visualize_class_analysis(
            class_name=args.class_name,
            save_path=save_path,
            top_k_atoms=10,
            show_images=5
        )

        # Visualize atom activations on specific image
        print("\nGenerating atom activation visualization on input image...")
        save_path = os.path.join(args.output_dir, f'{args.class_name}_atom_activations_img0.png')
        analyzer.visualize_atom_activations_on_image(
            class_name=args.class_name,
            image_idx=0,
            top_k_atoms=6,
            save_path=save_path
        )

        # Visualize atom activations across multiple images
        print("\nGenerating cross-image atom activation visualization...")
        save_path = os.path.join(args.output_dir, f'{args.class_name}_atoms_across_images.png')
        analyzer.visualize_cross_class_comparison(
            class_name=args.class_name,
            top_k_atoms=6,
            save_path=save_path
        )

    # Cross-class comparison
    else:
        print(f"\n{'='*70}")
        print(f"Cross-Class Comparison Mode")
        print(f"{'='*70}")

        for class_name in args.compare_classes:
            analyzer.analyze_class(
                class_name=class_name,
                num_images=args.num_images,
                cumulative_threshold=args.cumulative_threshold,
                lr_z=3.0,
                sparsity=0.005,
                n_steps_z=50
            )

        # Visualize multi-class comparison heatmap
        print("\nGenerating multi-class comparison...")
        save_path = os.path.join(args.output_dir, 'multi_class_comparison.png')
        analyzer.visualize_multi_class_comparison(
            class_names=args.compare_classes,
            save_path=save_path,
            top_k_atoms=20
        )

        # Also create individual class visualizations with atom activations
        for class_name in args.compare_classes:
            print(f"\nGenerating visualizations for '{class_name}'...")

            # Main analysis
            save_path = os.path.join(args.output_dir, f'{class_name}_analysis.png')
            analyzer.visualize_class_analysis(
                class_name=class_name,
                save_path=save_path,
                top_k_atoms=10,
                show_images=5
            )

            # Atom activations on image
            save_path = os.path.join(args.output_dir, f'{class_name}_atom_activations_img0.png')
            analyzer.visualize_atom_activations_on_image(
                class_name=class_name,
                image_idx=0,
                top_k_atoms=6,
                save_path=save_path
            )

            # Atoms across images
            save_path = os.path.join(args.output_dir, f'{class_name}_atoms_across_images.png')
            analyzer.visualize_cross_class_comparison(
                class_name=class_name,
                top_k_atoms=6,
                save_path=save_path
            )

    print(f"\n{'='*70}")
    print(f"✓ Analysis complete! Check '{args.output_dir}/' for results.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
