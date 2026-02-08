import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_samples(model, dataloader, device, num_samples=5, output_path='pytorch_gradcam_samples.png'):
    """
    Generates Grad-CAM samples for a few images in the dataloader.
    """
    model.eval()
    
    # Target layer for ResNet50 is usually the last bottleneck block of layer4
    target_layers = [model.layer4[-1]]
    
    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    images_processed = 0
    
    # De-normalization for visualization
    # Mean and Std from pytorch_dataset.get_transforms
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 5))
    if num_samples == 1:
        axs = [axs]
        
    iter_loader = iter(dataloader)
    
    try:
        images, labels = next(iter_loader)
    except StopIteration:
        print("Dataloader is empty.")
        return

    for i in range(min(num_samples, len(images))):
        input_tensor = images[i].unsqueeze(0).to(device)
        label_idx = labels[i].item()
        
        # We target the ground truth class
        targets = [ClassifierOutputTarget(label_idx)]
        
        # Generate CAM
        # Grayscale CAM is 1xHxW
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Prepare image for visualization
        # Un-normalize: input is (C, H, W), we need (H, W, C)
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        
        axs[i].imshow(visualization)
        axs[i].set_title(f"Class: {label_idx}")
        axs[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Grad-CAM samples saved to {output_path}")

