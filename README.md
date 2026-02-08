# LungHist700 Experiment & PyTorch Migration

This repository contains code for training and evaluating a ResNet50 model on the LungHist700 dataset. It includes both the original TensorFlow/Keras implementation (cpu-compatible) and a new, optimized PyTorch implementation (gpu-accelerated).

## 🚀 Quick Start (PyTorch)

The PyTorch implementation is recommended as it modernizes the codebase, resolves dependency issues, and successfully utilizes the GPU.

### 1. Setup Environment
```bash
# Install uv if not present
pip install uv

# Create venv and install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install torch torchvision torchmetrics grad-cam
```

### 2. Run Training
```bash
# Run for 10 epochs with batch size 8
python train_pytorch.py --epochs 10 --batch_size 8
```

Outputs will be saved to `pytorch_logs/`:
- `best_model.pth`: Best model checkpoint.
- `training_history.png`: Loss and Accuracy curves.
- `gradcam_samples.png`: Visualization of model attention.

---

## 📂 Project Structure

- **`train_pytorch.py`**: Main training script (PyTorch).
- **`run_experiment.py`**: Main training script (TensorFlow legacy).
- **`HistoLib/`**:
    - `pytorch_dataset.py`: PyTorch Dataset & DataLoader.
    - `pytorch_model.py`: ResNet50 model definition.
    - `pytorch_gradcam.py`: Grad-CAM visualization utilities.
    - `generator.py`: (Legacy) Keras data generators.
    - `models.py`: (Legacy) Keras model definition.
    - `traintest.py`: (Legacy) Keras training loop.

---

## 🛠 Changes & Improvements

### PyTorch Migration (Recommended)
We migrated the codebase to PyTorch to resolve system-level CUDA mismatch issues.
- **Benefits**:
    - **GPU Support**: Verified to work with CUDA.
    - **Modern Stack**: Removes dependency on deprecated `tensorflow-addons`.
    - **Eager Execution**: Easier debugging.

### TensorFlow Fixes (Legacy)
We also patched the original TensorFlow code to run on modern Python (3.13) environments (CPU-only):
- Removed `tensorflow-addons` dependencies.
- Updated `albumentations` usage (fixed `RandomSizedCrop` deprecation).
- Fixed `sklearn` compatibility issues in class weight computation.
- Fixed `ModelCheckpoint` file naming for Keras 3.

## 📊 Artifacts & Documentation
- **[Architecture Diagram](architecture_diagram.md)**: Logic flow of the original codebase.
- **[PyTorch Migration Evaluation](pytorch_migration_evaluation.md)**: Analysis of why migration was necessary.