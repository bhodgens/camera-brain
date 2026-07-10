# Vehicle Type Classifier Training - Google Colab

## Quick Start (Free GPU)

### Step 1: Open the Colab Notebook

1. Go to https://colab.research.google.com
2. Click **Upload** → **GitHub** tab
3. Paste this URL: `https://github.com/YOUR_USERNAME/rock-cluster/blob/main/training/vehicle-classifier/vehicle_classifier_training.ipynb`
   - OR: Upload the `.ipynb` file directly to Google Drive, then open from Drive
4. **Runtime** → **Change runtime type** → **GPU** (T4)

### Step 2: Download & Prepare Dataset

The notebook will automatically download:
- **Stanford Cars** (196 car models, ~16k images)
- **CompCars** subset (surveillance view only)

Or upload your own dataset in the format:
```
/data/
  ├── sedan/
  │   ├── img_001.jpg
  │   └── ...
  ├── SUV/
  ├── truck/
  ├── van/
  ├── pickup/
  ├── bus/
  ├── motorcycle/
  └── bicycle/
```

### Step 3: Run All Cells

Click **Runtime** → **Run all** and wait (~30-45 minutes on T4).

### Step 4: Download Trained Model

After training completes:
- `vehicle_classifier_best.pth` - PyTorch weights
- `vehicle_classifier.onnx` - ONNX format
- `config.json` - Class mappings

Download to: `/home/camera-brain/models/vehicle_classifier.rknn` (after RKNN conversion)

---

## Colab Free GPU Details

| GPU | VRAM | Session Limit | Notes |
|-----|------|---------------|-------|
| **NVIDIA T4** | 16GB | 12 hours | Most common, perfect for this task |

You get ~12 hours of continuous GPU time before session resets. Training takes ~45 minutes, so you have plenty of buffer.

## Training Output

```
Epoch 1/30 - Train Acc: 0.65 - Val Acc: 0.72
Epoch 2/30 - Train Acc: 0.78 - Val Acc: 0.81
...
Epoch 30/30 - Train Acc: 0.95 - Val Acc: 0.89

✓ Training complete!
✓ Best model saved: vehicle_classifier_best.pth
✓ ONNX exported: vehicle_classifier.onnx
```

Expected final accuracy: **88-92%** on validation set.
