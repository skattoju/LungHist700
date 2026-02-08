# PyTorch Migration Evaluation

This document evaluates the feasibility and benefits of migrating the `LungHist700` codebase from TensorFlow/Keras to PyTorch.

## Overview
The current codebase uses **TensorFlow (Keras)** with **ResNet50V2**. It relies on `tensorflow-addons` (deprecated) for some metrics and `albumentations` for augmentation.

## Comparison

| Feature | TensorFlow / Keras (Current) | PyTorch (Proposed) |
| :--- | :--- | :--- |
| **Model Definition** | High-level `keras.models.Model`. Concise. | `torch.nn.Module`. verbose but explicit. |
| **Data Loading** | `keras.utils.Sequence`. Flexible but boilerplate. | `torch.utils.data.Dataset` / `DataLoader`. Standard, optimized. |
| **Augmentation** | Functional integration via `albumentations`. | Native integration in `Dataset.__getitem__`. Very common. |
| **Training Loop** | `model.fit()`. Very easy, handles callbacks. | Explicit loop. More control, more boilerplate (unless using Lightning). |
| **Metrics** | `tf.keras.metrics` (some deprecated in Addons). | `torchmetrics` or `sklearn`. Robust ecosystem. |
| **Visualization** | `Grad-CAM` (custom implementation). | `Captum` or `torch-cam` libraries. |
| **Debugging** | Eager execution (modern TF). Generally harder stack traces. | Eager by default. Pythonic debugging. |

## Pros of Migration
1.  **Dependency Stability**: `tensorflow-addons` is deprecated and causing installation issues on modern Python (3.12+). PyTorch ecosystem is generally more stable regarding ancillary libraries.
2.  **Modern Ecosystem**: PyTorch is currently the dominant framework for research. Finding modern implementations of new techniques is easier.
3.  **Flexibility**: Easier to implement custom training steps or loss functions if the project evolves.
4.  **Debugging**: Pythonic nature makes debugging complex pipelines easier.

## Cons of Migration
1.  **Effort**: Requires rewriting the entire pipeline:
    -   Dataset/Dataloader (`generator.py` needs complete rewrite).
    -   Model wrapper (`models.py` needs porting ResNet50V2 to `torchvision` equivalent).
    -   Training loop (`traintest.py` needs complete rewrite using Torch or Lightning).
    -   Grad-CAM (`gradcam.py` needs rewrite or replacement).
2.  **Performance**: Keras `fit()` is highly optimized out-of-the-box. A poorly written PyTorch loop might be slower.
3.  **Weights**: Pre-trained Keras weights might need conversion if exact reproducibility of an existing checkpoint is needed (though ImageNet weights are available in both).

## Migration Plan (Rough Estimate: 1-2 Days)
1.  **Data Layer**: Wrap `albumentations` pipeline in a `LungHistDataset(Dataset)`. Use `DataLoader`.
2.  **Model**: Use `torchvision.models.resnet50(weights='IMAGENET...')`. Modify head (FC + Dropout).
3.  **Training**: Implement a standard training loop (optimizer, loss, backprop) or use **PyTorch Lightning** to keep it concise like Keras.
4.  **Metrics**: Use `torchmetrics` for F1, MCC, etc.
5.  **Grad-CAM**: Use `pytorch-grad-cam` library instead of custom code.

## Recommendation
**Recommend Migration.**
The deprecation of `tensorflow-addons` and version compatibility issues with TensorFlow on newer Python versions will continue to be a maintenance burden. Migrating to PyTorch (especially with PyTorch Lightning) will modernise the codebase, solve dependency hell, and make future extensions easier.
