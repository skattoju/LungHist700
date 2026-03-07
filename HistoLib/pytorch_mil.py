import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def get_mil_grid_augment_transforms():
    """Augmentation for pre-cropped 224x224 grid patches (no RandomCrop needed)."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GridDistortion(p=0.2),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_mil_grid_inference_transforms():
    """Normalize-only transform for pre-cropped 224x224 grid patches."""
    return A.Compose([
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
        
        # Note: We do NOT downscale the image for MIL. We want to extract 224x224 patches
        # straight from the original 1200x1600 high-resolution images to retain micro-texture.

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = imageio.imread(img_path)
        
        # No initial resize for MIL - retain true resolution
        
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

class MIL_GridFeatureDataset(Dataset):
    def __init__(self, image_paths, labels, class_names, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform
        self.patch_size = 224
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = imageio.imread(img_path)
        h, w, _ = image.shape
        
        # Calculate a 5x7 grid (35 patches) across the 1200x1600 image.
        y_steps = np.linspace(0, h - self.patch_size, 5).astype(int)
        x_steps = np.linspace(0, w - self.patch_size, 7).astype(int)
        
        patches = []
        for y in y_steps:
            for x in x_steps:
                crop = image[y:y+self.patch_size, x:x+self.patch_size]
                if self.transform:
                    augmented = self.transform(image=crop)
                    patch = augmented["image"]
                else:
                    patch = torch.from_numpy(crop.transpose(2, 0, 1)).float() / 255.0
                patches.append(patch)
                
        bag = torch.stack(patches, dim=0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Optionally return a patient/image id if needed for tracking 
        file_name = os.path.basename(img_path)
        return bag, label, file_name

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

def get_mil_grid_dataloaders(resolution='20x', batch_size=1, root_directory='data/images/', dataset_csv='data/data.csv', 
                             reproducible=True, num_workers=2):
    
    df = utils.get_dataframe(dataset_csv, resolution=resolution)
    class_names, labels = utils.get_classes_labels(root_directory, df['image_path'].values)
    df['targetclass'] = labels

    if resolution == '20x':
        train_ids = [2, 3, 4, 5, 7, 8, 12, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 28, 29, 30, 33, 36, 37, 38, 39, 41, 42, 45]
        val_ids = [1, 6, 27, 32, 44]
        test_ids = [9, 13, 31, 40]
    else:
        train_ids = [2, 6, 8, 9, 10, 12, 13, 14, 16, 18, 19, 21, 22, 24, 28, 29, 31, 33, 34, 35, 36, 38, 40, 44]
        val_ids = [1, 4, 17, 26, 30, 37, 45]
        test_ids = [11, 15, 20, 25, 32, 43]

    df_train = df[df.patient_id.isin(train_ids)]
    df_val = df[df.patient_id.isin(val_ids)]
    df_test = df[df.patient_id.isin(test_ids)]

    # Use inference transforms for all (since we are just extracting features from the raw image)
    transform = get_mil_transforms(is_train=False)

    train_ds = MIL_GridFeatureDataset(df_train['image_path'].values, df_train['targetclass'].values, class_names, transform=transform)
    val_ds = MIL_GridFeatureDataset(df_val['image_path'].values, df_val['targetclass'].values, class_names, transform=transform)
    test_ds = MIL_GridFeatureDataset(df_test['image_path'].values, df_test['targetclass'].values, class_names, transform=transform)

    # Batch size 1 because each item is a bag of 35 patches.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names

def get_frozen_resnet50():
    """Return a frozen ResNet50 feature extractor (no FC, no grad)."""
    resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    backbone = nn.Sequential(*list(resnet.children())[:-1])  # removes final FC -> output [B, 2048, 1, 1]
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def _extract_grid_patches(image, patch_size=224):
    """Extract a 5x7 grid of patches from a single image array."""
    h, w = image.shape[:2]
    y_steps = np.linspace(0, h - patch_size, 5).astype(int)
    x_steps = np.linspace(0, w - patch_size, 7).astype(int)
    patches = []
    for y in y_steps:
        for x in x_steps:
            patches.append(image[y:y + patch_size, x:x + patch_size])
    return patches


def extract_and_save_grid_embeddings(image_paths, labels, backbone, save_dir, device, transform=None, num_augmentations=0):
    """Extract embeddings for every image using a frozen backbone and save to disk.

    Args:
        image_paths: array of image file paths
        labels: array of integer labels (parallel to image_paths)
        backbone: frozen ResNet50 feature extractor
        save_dir: directory to write .pt files
        device: torch device
        transform: albumentations transform applied to each 224x224 patch
        num_augmentations: if >0, produce this many augmented copies per image
                           (saved as <name>_aug0.pt … <name>_augN.pt) in addition
                           to a single non-augmented version (<name>.pt).
    """
    os.makedirs(save_dir, exist_ok=True)
    backbone = backbone.to(device)

    inference_tf = get_mil_grid_inference_transforms()
    augment_tf = transform if transform is not None else get_mil_grid_augment_transforms()

    labels_dict = {}

    for i, img_path in enumerate(image_paths):
        image = imageio.imread(img_path)
        file_stem = os.path.splitext(os.path.basename(img_path))[0]
        labels_dict[file_stem] = int(labels[i])
        raw_patches = _extract_grid_patches(image)

        # --- non-augmented version (always saved) ---
        out_path = os.path.join(save_dir, f"{file_stem}.pt")
        if not os.path.exists(out_path):
            tensors = [inference_tf(image=p)["image"] for p in raw_patches]
            bag = torch.stack(tensors).to(device)  # [35, 3, 224, 224]
            with torch.no_grad():
                feats = backbone(bag).squeeze(-1).squeeze(-1)  # [35, 2048]
            torch.save(feats.cpu(), out_path)

        # --- augmented versions ---
        for aug_idx in range(num_augmentations):
            aug_path = os.path.join(save_dir, f"{file_stem}_aug{aug_idx}.pt")
            if not os.path.exists(aug_path):
                tensors = [augment_tf(image=p)["image"] for p in raw_patches]
                bag = torch.stack(tensors).to(device)
                with torch.no_grad():
                    feats = backbone(bag).squeeze(-1).squeeze(-1)
                torch.save(feats.cpu(), aug_path)

    return labels_dict


class MIL_EmbeddingDataset(Dataset):
    """Loads precomputed [num_patches, 2048] embeddings from disk.

    When num_augmentations > 0 the dataset randomly selects one of the
    augmented variants for each sample on every access (training-time
    augmentation via precomputed copies).
    """
    def __init__(self, embedding_dir, labels_dict, num_augmentations=0):
        self.embedding_dir = embedding_dir
        self.labels_dict = labels_dict
        self.num_augmentations = num_augmentations

        # Build the list of *base* file stems (exclude _aug* duplicates)
        self.stems = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(embedding_dir)
            if f.endswith('.pt') and '_aug' not in f
        ])

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]

        if self.num_augmentations > 0:
            aug_idx = np.random.randint(0, self.num_augmentations)
            file_name = f"{stem}_aug{aug_idx}.pt"
        else:
            file_name = f"{stem}.pt"

        file_path = os.path.join(self.embedding_dir, file_name)
        bag_embedding = torch.load(file_path, weights_only=True)
        label = torch.tensor(self.labels_dict[stem], dtype=torch.long)
        return bag_embedding, label


def get_embedding_dataloaders(embedding_base_dir, train_labels, val_labels, test_labels,
                              num_augmentations=5, batch_size=16, num_workers=2):
    """Create DataLoaders for precomputed embedding directories.

    Args:
        embedding_base_dir: parent directory containing train/, val/, test/ subdirs
        train_labels / val_labels / test_labels: dict mapping file stem -> int label
        num_augmentations: number of augmented copies available for training
        batch_size: batch size for all loaders
        num_workers: dataloader workers
    """
    train_ds = MIL_EmbeddingDataset(
        os.path.join(embedding_base_dir, 'train'), train_labels,
        num_augmentations=num_augmentations
    )
    val_ds = MIL_EmbeddingDataset(
        os.path.join(embedding_base_dir, 'val'), val_labels,
        num_augmentations=0
    )
    test_ds = MIL_EmbeddingDataset(
        os.path.join(embedding_base_dir, 'test'), test_labels,
        num_augmentations=0
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

class MIL_AttentionOnly(nn.Module):
    def __init__(self, num_classes, embed_dim=2048, num_heads=4):
        super(MIL_AttentionOnly, self).__init__()
        
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(self.embed_dim, num_classes)

    def forward(self, features):
        # features shape: [B, num_patches, 2048]
        # Multi-Head Attention (Self-attention)
        attn_output, _ = self.attention(features, features, features) # [B, num_patches, 2048]
        
        # Average pooling over patches to obtain a single embedding per bag
        pooled_output = torch.mean(attn_output, dim=1) # [B, 2048]
        
        # Classification
        logits = self.fc(pooled_output) # [B, num_classes]
        
        return logits


class GatedAttentionMIL(nn.Module):
    """Gated Attention MIL (Ilse et al. 2018).

    Computes per-instance attention weights via element-wise gating:
        a_k = softmax( W^T (tanh(V h_k) * sigmoid(U h_k)) )
    then aggregates:  z = sum_k a_k * h_k  ->  classifier(z)
    """
    def __init__(self, num_classes, embed_dim=2048, hidden_dim=256, dropout=0.25):
        super(GatedAttentionMIL, self).__init__()
        self.embed_dim = embed_dim

        self.attention_V = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_W = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, features):
        # features: [B, N, embed_dim]
        V = self.attention_V(features)   # [B, N, hidden_dim]
        U = self.attention_U(features)   # [B, N, hidden_dim]
        scores = self.attention_W(V * U) # [B, N, 1]
        attn_weights = F.softmax(scores, dim=1)  # [B, N, 1]

        # Weighted sum of instance embeddings
        z = torch.sum(attn_weights * features, dim=1)  # [B, embed_dim]
        logits = self.classifier(z)  # [B, num_classes]
        return logits
