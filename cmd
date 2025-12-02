 python explain_csae.py --image_path data/imagenette/gas_pump/ILSVRC2012_val_00000732.JPEG --cumulative_threshold 0.8 --top_k_features 10 --max_channels 5

  # Analyze single class
  python check_same_class_csae.py --class_name tench --num_images 10

  # Compare multiple classes
  python check_same_class_csae.py --compare_classes tench church --num_images 10
  python check_same_class_dual_csae.py --compare_classes tench church parachute --num_images 10 --use_resnet18 --dual_csae_path dual_csae_resnet18_model.pkl

  python visualize_multichannel_sae.py --image_path data/imagenette/tench/n01440764_1.JPEG

  # Comprehensive analysis on multiple images
  python visualize_multichannel_sae.py --class_name tench --num_images 10 --grid_view

   # Default view with activation maps
  python visualize_multichannel_sae.py \
    --image_path
  data/imagenette/gas_pump/ILSVRC2012_val_00004452.JPEG \
    --top_k_features 16

  # Grid view (detailed, 4 columns per feature)
  python visualize_multichannel_sae.py \
    --image_path
  data/imagenette/gas_pump/ILSVRC2012_val_00004452.JPEG \
    --grid_view \
    --top_k_features 16

  # Analyze multiple images from a class
  python visualize_multichannel_sae.py \
    --class_name tench \
    --num_images 5 \
    --top_k_features 16