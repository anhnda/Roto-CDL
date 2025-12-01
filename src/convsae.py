import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Model Definitions (Same as before)
# ==========================================

class ConvSAE(nn.Module):
    def __init__(self, in_channels=256, hidden_dim=4096, kernel_size=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.encoder = nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=padding)
        self.decoder = nn.Conv2d(hidden_dim, in_channels, kernel_size, padding=padding)
        self.encoder_bias = nn.Parameter(torch.zeros(hidden_dim))

        # Initialize encoder with Kaiming
        nn.init.kaiming_uniform_(self.encoder.weight)

        # Initialize decoder with positive weights (compatible with non-negativity constraint)
        # Use uniform distribution in [0, 0.02] to ensure all weights start positive
        nn.init.uniform_(self.decoder.weight, a=0.0, b=0.02)

    def forward(self, x):
        pre_act = self.encoder(x) + self.encoder_bias.view(1, -1, 1, 1)
        feature_acts = F.relu(pre_act)
        reconstruction = self.decoder(feature_acts)
        return reconstruction, feature_acts

    @torch.no_grad()
    def normalize_decoder_weights(self):
        """
        Normalize decoder weights with non-negativity constraint.
        Since feature_acts = ReLU(encoder(x)) are always non-negative,
        decoder weights should also be non-negative for meaningful reconstruction.
        """
        weight = self.decoder.weight
        # Apply ReLU to enforce non-negativity
        weight = torch.relu(weight)
        # L2 normalization
        norms = weight.norm(p=2, dim=(0, 2, 3), keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        self.decoder.weight.data = weight / norms

class LateralInhibitionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
        self.kernel = k.view(1, 1, 3, 3)
        self.padding = 1

    def forward(self, feature_map):
        b, c, h, w = feature_map.shape
        device = feature_map.device
        filters = self.kernel.repeat(c, 1, 1, 1).to(device)
        neighbor_sum = F.conv2d(feature_map, filters, padding=self.padding, groups=c)
        loss = (feature_map * neighbor_sum).mean()
        return loss


class ClassDiversityLoss(nn.Module):
    """
    Encourages different classes to activate different features (class-discriminative learning).

    This loss promotes orthogonality between class-specific feature activations while
    allowing some shared features for common patterns.

    The loss works by:
    1. Computing average feature activation for each class
    2. Measuring similarity between different classes' feature usage
    3. Penalizing high similarity (encourages class-specific features)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, feature_acts, labels):
        """
        Args:
            feature_acts: [B, C, H, W] - sparse feature activations
            labels: [B] - class labels (0 to num_classes-1)

        Returns:
            diversity_loss: scalar - penalty for feature overlap between classes
                          (0 = classes use different features, 1 = classes use same features)
        """
        B, C, H, W = feature_acts.shape
        device = feature_acts.device

        # Compute per-class feature activation strength
        # class_features[c, i] = average activation of feature i for class c
        class_features = torch.zeros(self.num_classes, C, device=device)

        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                # Average activation across spatial dims and samples
                class_acts = feature_acts[mask].mean(dim=(0, 2, 3))  # [C]
                class_features[c] = class_acts

        # Normalize feature vectors (L2 normalization)
        class_features_norm = F.normalize(class_features, dim=1, p=2, eps=1e-8)  # [num_classes, C]

        # Compute pairwise cosine similarity between classes
        similarity_matrix = torch.mm(class_features_norm, class_features_norm.t())  # [num_classes, num_classes]

        # Penalize off-diagonal elements (high similarity between different classes)
        # We want different classes to have LOW similarity (orthogonal features)
        mask = torch.ones_like(similarity_matrix) - torch.eye(self.num_classes, device=device)
        diversity_loss = (similarity_matrix.abs() * mask).sum() / (self.num_classes * (self.num_classes - 1))

        return diversity_loss

# ==========================================
# 2. Training with Detailed Logging
# ==========================================

# # Hyperparameters
# BATCH_SIZE = 32
# INPUT_CHANNELS = 256
# SPARSE_DIM = 4096
# LAMBDA_L1 = 0.005
# LAMBDA_LAT = 0.01
# LR = 3e-4
# EPOCHS = 20

# # Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ConvSAE(in_channels=INPUT_CHANNELS, hidden_dim=SPARSE_DIM).to(device)
# optimizer = optim.Adam(model.parameters(), lr=LR)
# lat_inhib_loss = LateralInhibitionLoss().to(device)

# # --- LOGGING DICTIONARY ---
# logs = {
#     "total_loss": [],
#     "recon_loss": [],
#     "l1_loss": [],
#     "lateral_loss": [],
#     "active_neurons_pct": [] # % of neurons that fired at least once in this batch
# }

# # Dummy Data (Replace with your AlexNet activations)
# data_loader = [torch.randn(BATCH_SIZE, INPUT_CHANNELS, 13, 13).to(device) for _ in range(50)]

# print("Starting Training with Diagnostics...")

# for epoch in range(EPOCHS):
#     epoch_recon = 0
#     epoch_l1 = 0
    
#     for batch_idx, batch_acts in enumerate(data_loader):
#         optimizer.zero_grad()
        
#         # Forward
#         recon, acts = model(batch_acts)
        
#         # Losses
#         loss_recon = F.mse_loss(recon, batch_acts)
#         loss_l1 = acts.abs().sum() / acts.numel() # Average activation value
#         loss_lat = lat_inhib_loss(acts)
        
#         loss = loss_recon + (LAMBDA_L1 * loss_l1) + (LAMBDA_LAT * loss_lat)
        
#         loss.backward()
#         optimizer.step()
#         model.normalize_decoder_weights()
        
#         # --- COLLECT METRICS ---
#         with torch.no_grad():
#             # Calculate what % of neurons are "Alive" (firing > 0) in this batch
#             # This is critical: if this drops to 0, your model is dead.
#             active_mask = (acts > 0).float()
#             active_pct = active_mask.mean().item() * 100 
            
#             logs["total_loss"].append(loss.item())
#             logs["recon_loss"].append(loss_recon.item())
#             logs["l1_loss"].append(loss_l1.item())
#             logs["lateral_loss"].append(loss_lat.item())
#             logs["active_neurons_pct"].append(active_pct)

#     print(f"Epoch {epoch+1}/{EPOCHS} | Recon: {loss_recon.item():.4f} | Active Neurons: {active_pct:.2f}%")

# ==========================================
# 3. Visualization Function
# ==========================================

def plot_training_logs(logs):
    """
    Plots a 2x2 Grid of health metrics.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('2D Conv-SAE Training Diagnostics', fontsize=16)
    
    # 1. Reconstruction (Should go down)
    axs[0, 0].plot(logs["recon_loss"], color='blue')
    axs[0, 0].set_title("Reconstruction Loss (MSE)")
    axs[0, 0].set_ylabel("Error")
    axs[0, 0].grid(True, alpha=0.3)
    
    # 2. Sparsity Level (Should stabilize near target)
    axs[0, 1].plot(logs["l1_loss"], color='green')
    axs[0, 1].set_title("L1 Loss (Avg Activation Magnitude)")
    axs[0, 1].set_ylabel("Magnitude")
    axs[0, 1].grid(True, alpha=0.3)
    
    # 3. Lateral Inhibition (Should go down)
    axs[1, 0].plot(logs["lateral_loss"], color='purple')
    axs[1, 0].set_title("Lateral Inhibition (Blob Penalty)")
    axs[1, 0].set_ylabel("Loss")
    axs[1, 0].grid(True, alpha=0.3)
    
    # 4. Active Neurons (Should NOT be 0)
    axs[1, 1].plot(logs["active_neurons_pct"], color='red')
    axs[1, 1].set_title("Active Neurons % (Batch-wise)")
    axs[1, 1].set_ylabel("Percent (%)")
    axs[1, 1].set_ylim(0, 100) # Fixed scale to see sparsity
    axs[1, 1].axhline(y=5, color='gray', linestyle='--', label='5% Target (Example)')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Run the plotter
#plot_training_logs(logs)