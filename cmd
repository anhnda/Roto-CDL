 python explain_csae.py --image_path data/imagenette/gas_pump/ILSVRC2012_val_00000732.JPEG --cumulative_threshold 0.8 --top_k_features 10 --max_channels 5

  # Analyze single class
  python check_same_class_csae.py --class_name tench --num_images 10

  # Compare multiple classes
  python check_same_class_csae.py --compare_classes tench church --num_images 10