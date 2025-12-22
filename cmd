# Train ResNet18                                                               
python run_xcsae_full.py --model resnet18                              
# Train VGG16                                                                  
python run_xcsae_full.py --model vgg16                                                                         
# Train EfficientNet                                                           
python run_xcsae_full.py --model efficientnet                                  
                                                                             
# ResNet18 early layer (higher resolution, fewer channels)                     
python run_xcsae_full.py --model resnet18 --target_layer layer2                
                                                                              
# VGG16 deeper layer (more semantic features)                                  
python run_xcsae_full.py --model vgg16 --target_layer features[23]     


                                                                                
  # Force resample test set                                                      
  python check_acc_drop_full.py --model resnet18 --force_resample                
                                                                                 
  # Custom batch size                                                            
  python check_acc_drop_full.py --model vgg16 --batch_size 64                    
                                                                                 
  # Custom output file                                                           
  python check_acc_drop_full.py \                                                
    --model efficientnet \                                                       
    --output_file my_results.txt    