# Federated Learning Feasibility Study

This document describes the Federated Learning (FL) simulation pipeline implemented for the LungHist700 dataset using PyTorch.

## Overview
We simulate a federated learning scenario where patient data is distributed across multiple clients (institutions). A global model is trained by aggregating local models trained on each client's private data, without sharing the raw data itself.

### Methodology
We follow a standard **Federated Averaging (FedAvg)** approach:
1.  **Data Partitioning**: The 45 patients in the dataset are partitioned into **3 disjoint buckets** (Clients). Each client has exclusive access to the images of its assigned patients.
2.  **Local Training**: Each client trains a local ResNet50V2 model on its own data for a specified number of epochs.
3.  **Aggregation**: The weights of the local models are sent to a central server, which computes the average weights:
    $$ W_{global} = \frac{1}{N} \sum_{i=1}^{N} W_{i} $$
4.  **Evaluation**: The aggregated global model is evaluated on a held-out test set (distinct from all training data) and compared against the individual local models.

## Implementation Details
The simulation is implemented in `run_fl_experiment_pytorch.py`.

### Components
-   **Patient Splitting**: Patients are shuffled and split into 3 equal-sized groups. All images belonging to a patient stay with that patient to ensure no leakage.
-   **Model**: We use a `ResNet50` backbone pretrained on ImageNet, with a custom classification head (Linear -> ReLU -> Dropout -> Linear).
-   **Optimization**:
    -   Loss: CrossEntropyLoss (weighted to handle class imbalance).
    -   Optimizer: Adam.
-   **Privacy**: In this simulation, data never leaves the client object. Only model weights are "shared" (via state_dicts).

## Running the Simulation
To run the simulation:
```bash
python run_fl_experiment_pytorch.py --epochs 10 --batch_size 8
```

### Options
-   `--resolution`: '20x' or '40x' (default: '20x')
-   `--epochs`: Number of local training epochs per round (default: 10)
-   `--seed`: Random seed for reproducibility (default: 42)
-   `--debug`: Run a fast 1-epoch test.

## Results
The script outputs:
-   **fl_pytorch_logs/fl_comparison_pytorch.png**: A bar chart comparing the test accuracy of each local model vs. the federated model.
-   **Console Output**: Loss and Accuracy metrics for each client and the federated model.
