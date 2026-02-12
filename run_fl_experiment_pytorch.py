import os
import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from HistoLib import pytorch_dataset, pytorch_model, utils

# --- Helper Functions ---

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def fed_avg(models_state_dicts):
    """
    Performs Federated Averaging on a list of state_dicts.
    """
    avg_state_dict = copy.deepcopy(models_state_dicts[0])
    for key in avg_state_dict.keys():
        for i in range(1, len(models_state_dicts)):
            avg_state_dict[key] += models_state_dicts[i][key]
        avg_state_dict[key] = torch.div(avg_state_dict[key], len(models_state_dicts))
    return avg_state_dict

# --- Main Simulation ---

def main():
    parser = argparse.ArgumentParser(description='Run Federated Learning Simulation (PyTorch)')
    parser.add_argument('--resolution', type=str, default='20x', choices=['20x', '40x'], help='Image resolution')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs per client')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='fl_pytorch_logs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Data Partitioning
    print("Partitioning data...")
    dataset_csv = 'data/data.csv'
    # Reuse utils to get dataframe and class names
    df = utils.get_dataframe(dataset_csv, resolution=args.resolution)
    class_names, labels = utils.get_classes_labels('data/images/', df['image_path'].values)
    df['targetclass'] = labels
    num_classes = len(class_names)
    
    # Get reproducible splits locally to avoid dependency issues with generator.py
    # Re-implementing simplified split logic using patient IDs directly
    if args.resolution == '20x':
        train_ids = [2, 3, 4, 5, 7, 8, 12, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 28, 29, 30, 33, 36, 37, 38, 39, 41, 42, 45]
        val_ids = [1, 6, 27, 32, 44]
        test_ids = [9, 13, 31, 40]
    else:
        train_ids = [2, 6, 8, 9, 10, 12, 13, 14, 16, 18, 19, 21, 22, 24, 28, 29, 31, 33, 34, 35, 36, 38, 40, 44]
        val_ids = [1, 4, 17, 26, 30, 37, 45]
        test_ids = [11, 15, 20, 25, 32, 43]

    df_train_global = df[df.patient_id.isin(train_ids)].copy()
    df_val_global = df[df.patient_id.isin(val_ids)].copy()
    df_test_global = df[df.patient_id.isin(test_ids)].copy()
    
    # Split Global Train patients into 3 Clients
    train_patients = df_train_global['patient_id'].unique()
    np.random.shuffle(train_patients)
    client_patient_splits = np.array_split(train_patients, 3)
    
    clients = []
    for i, p_ids in enumerate(client_patient_splits):
        client_df = df_train_global[df_train_global['patient_id'].isin(p_ids)]
        clients.append({
            'id': i,
            'patient_ids': p_ids,
            'df': client_df,
            'images': client_df['image_path'].values,
            'labels': client_df['targetclass'].values
        })
        print(f"Client {i}: {len(p_ids)} patients, {len(client_df)} images")

    # Global Validation and Test Loaders
    val_ds = pytorch_dataset.LungHistDataset(
        df_val_global['image_path'].values, 
        df_val_global['targetclass'].values, 
        class_names, 
        transform=pytorch_dataset.get_transforms(0.25, is_train=False)
    )
    test_ds = pytorch_dataset.LungHistDataset(
        df_test_global['image_path'].values, 
        df_test_global['targetclass'].values, 
        class_names, 
        transform=pytorch_dataset.get_transforms(0.25, is_train=False)
    )
    
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 2. Local Training
    client_state_dicts = []
    client_metrics_list = []
    
    for client in clients:
        print(f"\n--- Training Client {client['id']} ---")
        
        # Client Dataset & Loader
        client_ds = pytorch_dataset.LungHistDataset(
            client['images'], 
            client['labels'], 
            class_names, 
            transform=pytorch_dataset.get_transforms(0.25, is_train=True)
        )
        client_loader = DataLoader(client_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        # Initialize Model
        model = pytorch_model.get_model(num_classes).to(device)
        
        # Calculate Class Weights
        labels_np = client['labels']
        class_counts = np.bincount(labels_np, minlength=num_classes)
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1) 
        total_samples = len(labels_np)
        weights = total_samples / (num_classes * class_counts)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Training Loop
        history = {'loss': [], 'acc': []}
        epochs = 1 if args.debug else args.epochs
        
        for epoch in range(epochs):
            loss, acc = train_one_epoch(model, client_loader, criterion, optimizer, device)
            # Optional: Validate on global val set to monitor progress
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            
            history['loss'].append(loss)
            history['acc'].append(acc)
            print(f"Client {client['id']} Ep {epoch+1}: Loss {loss:.4f} Acc {acc:.4f} | Val Acc {val_acc:.4f}")
            
        # Save Weights
        client_state_dicts.append(copy.deepcopy(model.state_dict()))
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"client_{client['id']}_weights.pth"))
        
        # Default Evaluation (on Global Test Set)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        client_metrics_list.append({'id': client['id'], 'loss': test_loss, 'acc': test_acc})
        print(f"Client {client['id']} Test Acc: {test_acc:.4f}")
        
    # 3. Aggregation (FedAvg)
    print("\n--- Aggregating Models (FedAvg) ---")
    fed_state_dict = fed_avg(client_state_dicts)
    torch.save(fed_state_dict, os.path.join(args.output_dir, "federated_model_weights.pth"))
    
    # 4. Evaluation of Federated Model
    print("\n--- Evaluating Federated Model ---")
    fed_model = pytorch_model.get_model(num_classes).to(device)
    fed_model.load_state_dict(fed_state_dict)
    
    # Use unweighted criterion for fair test set comparison?
    # Usually test set metric is just accuracy, loss is secondary.
    # We'll use simple CE without weights for test loss report, or keep proper weights if test set is imbalanced.
    # Let's use unweighted for test loss consistency if we don't recalculate weights for test.
    criterion_test = nn.CrossEntropyLoss().to(device)
    
    fed_loss, fed_acc = evaluate(fed_model, test_loader, criterion_test, device)
    print(f"Federated Model Test Accuracy: {fed_acc:.4f}")
    
    # Comparison Output
    print("\nXXX RESULTS XXX")
    for m in client_metrics_list:
        print(f"Client {m['id']}: Loss={m['loss']:.4f}, Accuracy={m['acc']:.4f}")
    print(f"Federated: Loss={fed_loss:.4f}, Accuracy={fed_acc:.4f}")
    
    # Plot Comparison
    plt.figure(figsize=(10, 6))
    labels_plot = [f'Client {m["id"]}' for m in client_metrics_list] + ['Federated']
    accuracies = [m['acc'] for m in client_metrics_list] + [fed_acc]
    
    plt.bar(labels_plot, accuracies, color=['blue', 'blue', 'blue', 'green'])
    plt.title('Model Comparison: Local vs Federated (PyTorch)')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1.1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
    plt.savefig(os.path.join(args.output_dir, 'fl_comparison_pytorch.png'))
    print(f"Comparison plot saved to {os.path.join(args.output_dir, 'fl_comparison_pytorch.png')}")

if __name__ == "__main__":
    main()
