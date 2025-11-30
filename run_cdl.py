from src.activation import ActivationMapCollector
from src.model import FineTunedModel
from src.batch_cdl_large import roto_cdl_large_scale, vis_large_dict
import torch
from torch.utils.data import DataLoader
import joblib
# %%
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
from PIL import Image
if __name__ == "__main__":
    # 1. SETUP DATA
    print("Setting up Data...")
    data_dir = 'data/imagenette'
    BATCH_SIZE = 1 # Keep 1 for collection to save VRAM

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
    data_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE)

    # 2. COLLECT FEATURE MAPS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = FineTunedModel(num_classes=10).to(device)
    model.load_state_dict(torch.load('weights/finetune_weights.pth'))
    target_layer = model.feature_extractor[5]

    collector = ActivationMapCollector(model, target_layer, device=device)
    
    # Collect data (limit to first 20k if needed, or full dataset)
    print("Collecting Activation Maps...")
    X = collector.collect_maps(data_loader, output_tensor=True, device=device)
    # X = X[:5000] # Uncomment for quick testing
    
    # 3. ROBUST NORMALIZATION
    flat = X.flatten()
    num = min(10_000_000, flat.numel())
    idx = torch.randint(0, flat.numel(), (num,), device=flat.device)
    scale_factor = torch.quantile(flat[idx], 0.99)

    print(f"Robust Scale Factor: {scale_factor:.4f}")
    X = torch.clamp(X, min=0.0, max=scale_factor)
    X = X / scale_factor
    X = X * 10
    
    # OPTIONAL: BLURRING (Uncomment if "single dot" problem persists)
    # This forces atoms to learn shapes by making the input less spiky
    blur = transforms.GaussianBlur(kernel_size=3, sigma=0.5)
    X = blur(X)

    # 4. RUN DICTIONARY LEARNING
    print("Starting CDL Training...")
    phi_learned = roto_cdl_large_scale(
        X, 
        n_atoms=64,      # REDUCED from 400 to force feature sharing
        n_rotations=4, 
        d=9,             # Slightly larger kernel
        n_epochs=10, 
        batch_size=1500,
        
        # --- FINAL TUNED PARAMETERS ---
        lr_phi=0.05,
        lr_z=3.0,        # HIGH LR: Essential for Z to escape 0
        sparsity=0.005,  # MODERATE: Clean noise but keep signal
        # ------------------------------
        
        device=device
    )

    # 5. SAVE & VISUALIZE
    vis_large_dict(phi_learned)
    joblib.dump(phi_learned.detach().cpu(), 'phi_learned_imagenette.pkl')
    print("Done. Saved to phi_learned_imagenette.pkl and large_dictionary.png")