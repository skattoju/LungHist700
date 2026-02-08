import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, F1Score, AUROC, Precision, Recall, ConfusionMatrix
from HistoLib import pytorch_dataset, pytorch_model, pytorch_gradcam, utils

def train_one_epoch(model, loader, criterion, optimizer, device, metrics):
    model.train()
    running_loss = 0.0
    
    # Reset metrics
    for metric in metrics.values():
        metric.reset()

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Update metrics
        # CrossEntropyLoss expects class indices, inputs to metrics should be (preds, target)
        # outputs are logits, labels are indices
        for metric in metrics.values():
            metric(outputs, labels)
            
        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(loader)
    epoch_metrics = {name: metric.compute().item() for name, metric in metrics.items()}
    return epoch_loss, epoch_metrics

def validate(model, loader, criterion, device, metrics):
    model.eval()
    running_loss = 0.0
    
    for metric in metrics.values():
        metric.reset()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            for metric in metrics.values():
                metric(outputs, labels)

    epoch_loss = running_loss / len(loader)
    epoch_metrics = {name: metric.compute().item() for name, metric in metrics.items()}
    return epoch_loss, epoch_metrics

def main():
    parser = argparse.ArgumentParser(description='Train LungHist700 with PyTorch')
    parser.add_argument('--resolution', type=str, default='20x', choices=['20x', '40x'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output_dir', type=str, default='pytorch_logs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # DataLoaders
    train_loader, val_loader, test_loader, class_names = pytorch_dataset.get_dataloaders(
        resolution=args.resolution,
        batch_size=args.batch_size,
        image_scale=0.25, # Consistent with baseline
        reproducible=True
    )
    
    num_classes = len(class_names)
    
    if args.debug:
        # Limit data for debugging
        print("Debug mode: limiting epochs to 1 and batches")
        args.epochs = 1
    
    # Model
    model = pytorch_model.get_model(num_classes).to(device)
    
    # Class Weights
    # Using utils.compute_weights behavior: it returns a dict. We need a tensor.
    # Note: we can't easily use the generator-based compute_weights from utils because it expects a Keras sequence.
    # We will compute weights manually from the dataset logic or labels.
    # Quick way: collect all labels from train_loader (might be slow) or just assume balanced/imbalanced from knowns.
    # Or reimplement weight computation.
    # Let's grab labels from the dataset directly since we have access.
    train_labels = train_loader.dataset.labels
    # labels are numpy array of strings or ints? 
    # utils.get_classes_labels -> labels are integer encoded in the DF as 'targetclass'
    # but let's double check. pytorch_dataset uses df['targetclass'].values which are mapped integers.
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    # Balanced weights: total / (num_classes * count)
    weights = total_samples / (num_classes * class_counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Class weights: {weights}")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)

    # Metrics
    metrics_collection = {
        'acc': Accuracy(task="multiclass", num_classes=num_classes).to(device),
        'f1': F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device),
        #'auc': AUROC(task="multiclass", num_classes=num_classes).to(device) # Needs probabilities, might be tricky with logits? Torchmetrics handles it.
        # AUROC handles logits if task is multiclass.
    }

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("Starting training...")
    for epoch in range(args.epochs):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, metrics_collection)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, metrics_collection)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['acc'])
        history['val_acc'].append(val_metrics['acc'])
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} "
              f"Train Loss: {train_loss:.4f} Acc: {train_metrics['acc']:.4f} "
              f"Val Loss: {val_loss:.4f} Acc: {val_metrics['acc']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print("  Saved best model.")

    print("Training complete.")
    
    # Testing
    print("Evaluating on Test set...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    test_loss, test_metrics = validate(model, test_loader, criterion, device, metrics_collection)
    print(f"Test Loss: {test_loss:.4f} Test Acc: {test_metrics['acc']:.4f} Test F1: {test_metrics['f1']:.4f}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
    print("Saved training history plot.")

    # Grad-CAM Visualization
    if os.path.exists(os.path.join(args.output_dir, 'best_model.pth')):
        print("Generating Grad-CAM samples...")
        pytorch_gradcam.generate_samples(
            model, 
            test_loader, 
            device, 
            num_samples=5, 
            output_path=os.path.join(args.output_dir, 'gradcam_samples.png')
        )


if __name__ == "__main__":
    main()
