#!/usr/bin/env python3
"""
Script tiền xử lý dataset Imagenette
- Tải dataset từ Kaggle
- Giải nén
- Gộp train/val và đổi tên thư mục theo class name
"""

import os
import shutil
import subprocess
import kagglehub


def run_command(cmd):
    """Chạy shell command"""
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def count_images(root):
    """Đếm số lượng ảnh trong thư mục"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    count = 0

    for dirpath, dirnames, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f.lower())[1] in exts:
                count += 1

    return count


def main():
    # 1. Tạo thư mục data
    print("=== Bước 1: Tạo thư mục data ===")
    os.makedirs("data", exist_ok=True)

    # 2. Download dataset từ Kaggle
    print("\n=== Bước 2: Download dataset ===")
    path = kagglehub.dataset_download("jhoward/imagenette-160-px")
    print(f"Path to dataset files: {path}")
    
    # Di chuyển dataset
    run_command(f"mv {path} ./data/imagenette")

    # 3. Giải nén
    print("\n=== Bước 3: Giải nén dataset ===")
    run_command("tar -xvf data/imagenette/imagenette-160.tgz -C data/imagenette --strip-components 1")

    # 4. Đếm số ảnh trong train
    print("\n=== Bước 4: Đếm số ảnh ===")
    root_folder = "data/imagenette/train"
    print(f"Số lượng ảnh: {count_images(root_folder)}")

    # 5. Gộp và đổi tên thư mục
    print("\n=== Bước 5: Gộp train/val và đổi tên ===")
    
    # Ánh xạ từ mã ID sang tên lớp
    class_map = {
        'n01440764': 'tench',
        'n02102040': 'English_springer',
        'n02979186': 'cassette_player',
        'n03000684': 'chain_saw',
        'n03028079': 'church',
        'n03394916': 'French_horn',
        'n03417042': 'garbage_truck',
        'n03425413': 'gas_pump',
        'n03445777': 'golf_ball',
        'n03888257': 'parachute'
    }

    # Định nghĩa các đường dẫn
    base_dir = 'data/imagenette'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    print(f"Bắt đầu gộp và đổi tên tại: {base_dir}")

    # Lặp qua từng lớp để gộp và đổi tên
    for class_code, class_name in class_map.items():
        
        # Đường dẫn thư mục nguồn (train và val)
        source_train = os.path.join(train_dir, class_code)
        source_val = os.path.join(val_dir, class_code)
        
        # Đường dẫn thư mục đích (đã gộp)
        target_dir = os.path.join(base_dir, class_name)
        
        # Tạo thư mục đích nếu nó chưa tồn tại
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy từ 'train'
        if os.path.isdir(source_train):
            for filename in os.listdir(source_train):
                source_file = os.path.join(source_train, filename)
                target_file = os.path.join(target_dir, filename)
                shutil.copy(source_file, target_file)
        
        # Copy từ 'val'
        if os.path.isdir(source_val):
            for filename in os.listdir(source_val):
                source_file = os.path.join(source_val, filename)
                target_file = os.path.join(target_dir, filename)
                shutil.copy(source_file, target_file)
                
        print(f"[XONG] Đã gộp {class_code} -> {class_name}")

    # 6. Di chuyển thư mục train/val gốc
    print("\n=== Bước 6: Lưu trữ thư mục gốc ===")
    os.makedirs("data/original", exist_ok=True)
    run_command("mv data/imagenette/train data/original/")
    run_command("mv data/imagenette/val data/original/")

    # 7. Xóa file nén
    print("\n=== Bước 7: Xóa file nén ===")
    run_command("rm -rf data/imagenette/imagenette-160.tgz")

    print("\n✅ Hoàn tất! Tất cả các ảnh đã được gộp và đổi tên.")


if __name__ == "__main__":
    main()