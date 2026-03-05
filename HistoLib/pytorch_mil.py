import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio.v3 as imageio
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
from torchvision.models import ResNet50_Weights
from . import utils

def get_mil_transforms(target_size=224, is_train=True):
    if is_train:
        return A.Compose([
            A.RandomCrop(height=target_size, width=target_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GridDistortion(p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        # Evaluate with random crops as well, or we could do deterministic grids. 
        # The paper says "extracted 20 random patches of size 224x224".
        return A.Compose([
            A.RandomCrop(height=target_size, width=target_size, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

class MIL_LungHistDataset(Dataset):
    def __init__(self, image_paths, labels, class_names, num_patches=20, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.num_patches = num_patches
        self.transform = transform
        
        # We need an initial resize before cropping to match the baseline 0.25 scale 
        # Baseline resizes to 1200*0.25=300, 1600*0.25=400.
        self.base_resize = A.Resize(300, 400)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = imageio.imread(img_path)
        
        # Initial resize
        image = self.base_resize(image=image)["image"]
        
        patches = []
        for _ in range(self.num_patches):
            if self.transform:
                augmented = self.transform(image=image)
                patch = augmented["image"]
            else:
                # Fallback if no specific patch transform
                patch = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            patches.append(patch)
            
        # Stack patches into [num_patches, C, H, W]
        bag = torch.stack(patches, dim=0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return bag, label

def get_mil_dataloaders(resolution='20x', batch_size=3, root_directory='data/images/', dataset_csv='data/data.csv', 
                        train_split=0.8, val_split=0.1, random_state=17, num_workers=2, image_scale=0.25, reproducible=True):
    
    # Reuse utils logic
    df = utils.get_dataframe(dataset_csv, resolution=resolution)
    class_names, labels = utils.get_classes_labels(root_directory, df['image_path'].values)
    df['targetclass'] = labels

    if resolution == '20x':
        train_ids = [2, 3, 4, 5, 7, 8, 12, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 28, 29, 30, 33, 36, 37, 38, 39, 41, 42, 45]
        val_ids = [1, 6, 27, 32, 44]
        test_ids = [9, 13, 31, 40]
    else:
        # 40x ids
        train_ids = [2, 6, 8, 9, 10, 12, 13, 14, 16, 18, 19, 21, 22, 24, 28, 29, 31, 33, 34, 35, 36, 38, 40, 44]
        val_ids = [1, 4, 17, 26, 30, 37, 45]
        test_ids = [11, 15, 20, 25, 32, 43]

    df_train = df[df.patient_id.isin(train_ids)]
    df_val = df[df.patient_id.isin(val_ids)]
    df_test = df[df.patient_id.isin(test_ids)]

    train_ds = MIL_LungHistDataset(
        df_train['image_path'].values, df_train['targetclass'].values, class_names,
        num_patches=20, transform=get_mil_transforms(is_train=True)
    )
    
    val_ds = MIL_LungHistDataset(
        df_val['image_path'].values, df_val['targetclass'].values, class_names,
        num_patches=20, transform=get_mil_transforms(is_train=False)
    )
    
    test_ds = MIL_LungHistDataset(
        df_test['image_path'].values, df_test['targetclass'].values, class_names,
        num_patches=20, transform=get_mil_transforms(is_train=False)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names

class MIL_ResNet50(nn.Module):
    def __init__(self, num_classes, num_heads=4, freeze_base=False):
        super(MIL_ResNet50, self).__init__()
        # Backbone (ResNet50 without fc)
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        if freeze_base:
            for param in resnet.parameters():
                param.requires_grad = False
                
        # We need the pooled output (2048-d) per patch.
        # list(resnet.children())[:-1] gets up to AdaptiveAvgPool2d, outputting (B, 2048, 1, 1) per patch
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.embed_dim = resnet.fc.in_features # 2048
        
        # Attention Layer
        # The paper: "An attention layer with four heads was then applied"
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True)
        
        # Classification Head
        self.fc = nn.Linear(self.embed_dim, num_classes)

    def forward(self, bags):
        # bags shape: [B, num_patches, C, H, W]
        B, num_patches, C, H, W = bags.size()
        
        # Flatten batch and patches to put through standard ResNet50
        bags_flat = bags.view(B * num_patches, C, H, W)
        
        # Extract features
        features = self.backbone(bags_flat) # [B*20, 2048, 1, 1]
        features = features.view(B * num_patches, -1) # [B*20, 2048]
        
        # Reshape for Attention
        features = features.view(B, num_patches, self.embed_dim) # [B, 20, 2048]
        
        # Multi-Head Attention (Self-attention)
        attn_output, _ = self.attention(features, features, features) # [B, 20, 2048]
        
        # Average pooling over the 20 patches to obtain a single embedding per bag
        pooled_output = torch.mean(attn_output, dim=1) # [B, 2048]
        
        # Classification
        logits = self.fc(pooled_output) # [B, num_classes]
        
        return logits
