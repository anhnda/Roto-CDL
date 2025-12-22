python run_vgg_mask.py 
python  visualize_multichannel_vgg_mask.py --class_name golf_ball --num_images 10
python check_accuracy_drop_vgg_mask.py

                                                                         
  # First run: Extract 5 test images per class, visualize 10 random       
  samples                                                                 
  python visualize_testmf_resnet.py --num_samples 10 --top_k_features 16  
                                                                          
  # Customize test sampling                                               
  python visualize_testmf_resnet.py \                                     
    --test_images_per_class 5 \                                           
    --num_samples 20 \                                                    
    --top_k_features 12                                                   
                                                                          
  # Force resample test set                                               
  python visualize_testmf_resnet.py --force_resample                      
  --test_images_per_class 10                                              
                                                                          
  # Use different model                                                   
  python visualize_testmf_resnet.py \                                     
    --csae_model my_custom_model.pkl \                                    
    --num_samples 15                                                      
                                                                          
  Output                                                                  
                                                                          
  - Cached test samples: /data/imagenet1k_sampletest/test_metadata.pkl    
  - Visualizations:                                                       
  imagenet1k_test_visualizations/test_sample_X_labelY.png                 
  - Summary: Accuracy on visualized samples (e.g., "7/10 (70%)")          
                                                                          
  Configuration Parameters                                                
                                                                          
  - --test_images_per_class: Images to sample per class (default: 5)      
  - --num_samples: Number of random test images to visualize (default: 10)
  - --top_k_features: Features to show per image (default: 16)            
  - --force_resample: Force resampling from parquet files                 
  - --output_dir: Where to save visualizations                            
                                                                          
  The script follows the same structure as the training pipeline but works
   on the validation set for testing!                                     
                                     