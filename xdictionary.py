# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
from PIL import Image

# %%
from src.gradcam import GradCAM
from src.activation import ActivationMapCollector
from src.model import FineTunedModel


# %%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ----------------------------------------------------
# 1. ĐỊNH NGHĨA ĐƯỜNG DẪN VÀ BATCH SIZE
# ----------------------------------------------------
# Lấy đường dẫn từ ảnh của bạn
data_dir = 'data/imagenette' 
BATCH_SIZE = 1

# ----------------------------------------------------
# 2. ĐỊNH NGHĨA CÁC PHÉP BIẾN ĐỔI (TRANSFORMS)
# ----------------------------------------------------
# Đây là bước rất quan trọng.
# Model của bạn (giống AlexNet) cần ảnh đầu vào có kích thước
# cố định (ví dụ: 224x224) và đã được chuẩn hóa.

data_transform = transforms.Compose([
    # Resize ảnh về kích thước 224x224
    transforms.Resize((224, 224)),
    
    # (Tùy chọn) Thêm Augmentation để model học tốt hơn
    # transforms.RandomHorizontalFlip(), # Lật ảnh ngẫu nhiên
    
    # Chuyển ảnh (PIL Image) sang Tensor (PyTorch)
    # và scale giá trị pixel từ [0, 255] về [0.0, 1.0]
    transforms.ToTensor(),
    
    # Chuẩn hóa ảnh với Mean và Std của ImageNet
    # (Rất quan trọng nếu bạn dùng pre-trained weights)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------------------------------
# 3. TẠO DATASET BẰNG IMAGEFOLDER
# ----------------------------------------------------
# Đây là "phép thuật" - nó tự động tìm các lớp
# từ tên thư mục con.
full_dataset = datasets.ImageFolder(
    root=data_dir,
    transform=data_transform
)

print(f"Tìm thấy {len(full_dataset)} ảnh trong {len(full_dataset.classes)} lớp.")
print("Các lớp được tìm thấy:", full_dataset.classes)

# ----------------------------------------------------
# 4. TẠO DATALOADER
# ----------------------------------------------------
# DataLoader chịu trách nhiệm xáo trộn (shuffle),
# tạo batch, và tải dữ liệu song song.

data_loader = DataLoader(
    full_dataset,
    batch_size=BATCH_SIZE,

)

# ----------------------------------------------------
# 5. (TÙY CHỌN) KIỂM TRA
# ----------------------------------------------------
print("\nKiểm tra một batch từ DataLoader:")
try:
    # Lấy một batch đầu tiên
    images, labels = next(iter(data_loader))
    
    print(f"- Kích thước batch ảnh (Images shape): {images.shape}") 
    # Sẽ in ra: [64, 3, 224, 224] (Batch, Channels, Height, Width)
    
    print(f"- Kích thước batch nhãn (Labels shape): {labels.shape}") 
    # Sẽ in ra: [64]
    
except Exception as e:
    print(f"Lỗi khi tải dữ liệu: {e}")

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FineTunedModel(num_classes=10).to(device)
model.load_state_dict(torch.load('weights/finetune_weights.pth'))
target_layer = model.feature_extractor[5]


# %%
#print(model)

# %%
collector = ActivationMapCollector(model, target_layer, device=device)
X = collector.collect_maps(data_loader, output_tensor=True, device=device)
#X = X[:20000]
flat = X.flatten()
num = min(10_000_000, flat.numel())
idx = torch.randint(0, flat.numel(), (num,), device=flat.device)
scale_factor = torch.quantile(flat[idx], 0.99)


print(f"Robust Scale Factor: {scale_factor:.4f}")
X = torch.clamp(X, min=0.0, max=scale_factor)
X = X / scale_factor
X = X * 10
print("Shape X: ", X.shape)
# %%
import importlib
import src.batch_cdl_large
importlib.reload(src.batch_cdl_large)
from src.batch_cdl_large import roto_cdl_large_scale, vis_large_dict

# %%

phi_learned = roto_cdl_large_scale(
    X, 
    n_atoms=64,    # Large number of atoms
    n_rotations=4, 
    d=9,
    n_epochs=20, 
    batch_size=1500,
    
    # --- TUNED PARAMETERS ---
    lr_phi=0.05,
    lr_z=3,        # Aggressive inner loop to escape zero
    sparsity=0.0002, # Low target sparsity for high atom count
    # ------------------------
    
    device=device
)
vis_large_dict(phi_learned)
import joblib
joblib.dump(phi_learned.detach().cpu(), 'phi_learned_imagenette.pkl')
