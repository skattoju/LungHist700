import random
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_samples(model, dataloader, device, num_samples=5, output_path='pytorch_gradcam_samples.png', class_names=None):
    """
    Generates Grad-CAM samples for a few images in the dataloader.
    Works with both standard ResNet50 and MIL_ResNet50 models.
    """
    model.eval()
    
    # Detect target layer depending on model type.
    # MIL_ResNet50 wraps the backbone in self.backbone (nn.Sequential).
    # Standard ResNet50 exposes layer4 directly.
    if hasattr(model, 'layer4'):
        # Standard ResNet-based model
        target_layers = [model.layer4[-1]]
        is_mil = False
    elif hasattr(model, 'backbone'):
        # MIL_ResNet50: backbone is nn.Sequential(*list(resnet.children())[:-1])
        # Children order: conv1(0), bn1(1), relu(2), maxpool(3),
        #                 layer1(4), layer2(5), layer3(6), layer4(7), avgpool(8)
        target_layers = [model.backbone[7][-1]]
        is_mil = True
    else:
        raise AttributeError("Cannot determine Grad-CAM target layer for this model.")
    
    # For MIL, wrap the model so GradCAM sees a standard [B, C, H, W] -> logits interface
    if is_mil:
        class MILSinglePatchWrapper(torch.nn.Module):
            def __init__(self, mil_model):
                super().__init__()
                self.mil_model = mil_model
            def forward(self, x):
                # x: [1, C, H, W] — treat as a single-patch bag
                return self.mil_model(x.unsqueeze(1))  # [1, 1, C, H, W]
        cam_model = MILSinglePatchWrapper(model)
    else:
        cam_model = model

    # Initialize GradCAM
    cam = GradCAM(model=cam_model, target_layers=target_layers)
    
    # De-normalization for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 5))
    if num_samples == 1:
        axs = [axs]

    # Get a batch
    iter_loader = iter(dataloader)
    try:
        images, labels = next(iter_loader)
    except StopIteration:
        print("Dataloader is empty.")
        return
        
    # Shuffle within the batch to get varied classes (val_loader is not shuffled)
    batch_size = images.size(0)
    indices = list(range(batch_size))
    random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]

    for i in range(min(num_samples, len(images))):
        label_idx = labels[i].item()
        targets = [ClassifierOutputTarget(label_idx)]

        if is_mil:
            # images[i] shape: [num_patches, C, H, W] — pick first patch for visualization
            patch = images[i][0]  # [C, H, W]
            input_tensor = patch.unsqueeze(0).to(device)  # [1, C, H, W]
        else:
            input_tensor = images[i].unsqueeze(0).to(device)  # [1, C, H, W]
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Un-normalize for visualization
        img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        
        axs[i].imshow(visualization)
        display_name = class_names[label_idx] if class_names else str(label_idx)
        axs[i].set_title(f"Class: {display_name}")
        axs[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Grad-CAM samples saved to {output_path}")
