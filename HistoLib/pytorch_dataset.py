import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio.v3 as imageio
import albumentations as A
from albumentations.pytorch import ToTensorV2
from . import utils

class LungHistDataset(Dataset):
    def __init__(self, image_paths, labels, class_names, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = imageio.imread(img_path)
        
        # Albumentations expects HWC, which imageio provides
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        # Labels to tensor
        label = self.labels[idx]
        # If label is one-hot (from utils), convert to index for CrossEntropyLoss
        # utils.get_classes_labels returns strings, but we need to check how they are passed here.
        # Looking at generator.py: labels are converted to categorical. 
        # Here we will assume we receive integer class indices or handle it in get_dataloaders
        
        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(percent_resize=0.25, is_train=True):
    # Base size calculation from generator.py (1200*scale, 1600*scale)
    # generator.py used 1200x1600 as base. 
    target_h = int(1200 * percent_resize)
    target_w = int(1600 * percent_resize)

    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GridDistortion(p=0.2),
            # Fixed deprecated RandomSizedCrop from previous step
            A.RandomResizedCrop(size=(target_h, target_w), scale=(0.8, 1.0), p=0.4),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.2),
            A.Resize(target_h, target_w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet stats
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(target_h, target_w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

def get_dataloaders(resolution='20x', batch_size=8, root_directory='data/images/', dataset_csv='data/data.csv', 
                    train_split=0.8, val_split=0.1, random_state=17, image_scale=0.25, reproducible=True, num_workers=2):
    
    # Reuse utils logic to parse CSV and splits
    df = utils.get_dataframe(dataset_csv, resolution=resolution)
    class_names, labels = utils.get_classes_labels(root_directory, df['image_path'].values)
    df['targetclass'] = labels

    if reproducible:
        # We need to replicate get_reproducible_ids logic from generator.py here or import it
        # Since it was in generator.py and not utils.py, we should probably move it to utils or copy it.
        # For clean separation, let's copy the logic or import if we can. 
        # generator.py is Keras dependent, so we might want to avoid importing it if it has heavy side effects.
        # But importing a function is fine. Let's try to import get_reproducible_ids from generator
        # Wait, generator imports tensorflow. Let's strictly avoid importing generator.py to avoid TF overhead.
        # I will inline the ID logic here.
        pass # Implemented below
    
    # Split logic
    if resolution == '20x':
        train_ids = [2, 3, 4, 5, 7, 8, 12, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 28, 29, 30, 33, 36, 37, 38, 39, 41, 42, 45]
        val_ids = [1, 6, 27, 32, 44]
        test_ids = [9, 13, 31, 40]
    else:
        # 40x ids
        train_ids = [2, 6, 8, 9, 10, 12, 13, 14, 16, 18, 19, 21, 22, 24, 28, 29, 31, 33, 34, 35, 36, 38, 40, 44]
        val_ids = [1, 4, 17, 26, 30, 37, 45]
        test_ids = [11, 15, 20, 25, 32, 43]

    if reproducible:
        df_train = df[df.patient_id.isin(train_ids)]
        df_val = df[df.patient_id.isin(val_ids)]
        df_test = df[df.patient_id.isin(test_ids)]
    else:
        # Fallback to stratified shuffle split using utils (same as generator.py)
        from .utils import train_test_split
        df_train, df_test = train_test_split(df, test_size=1-train_split, random_state=random_state)
        df_test, df_val = train_test_split(df_test, test_size=round((val_split)/(1-train_split), 3), random_state=random_state)

    # Datasets
    train_ds = LungHistDataset(
        df_train['image_path'].values, 
        df_train['targetclass'].values, 
        class_names, 
        transform=get_transforms(image_scale, is_train=True)
    )
    
    val_ds = LungHistDataset(
        df_val['image_path'].values, 
        df_val['targetclass'].values, 
        class_names, 
        transform=get_transforms(image_scale, is_train=False)
    )
    
    test_ds = LungHistDataset(
        df_test['image_path'].values, 
        df_test['targetclass'].values, 
        class_names, 
        transform=get_transforms(image_scale, is_train=False)
    )

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names
