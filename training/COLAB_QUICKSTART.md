# Google Colab Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Open Colab
Go to: **https://colab.research.google.com**

### Step 2: Upload Notebook
1. Click **Upload** tab
2. Drag and drop one of these files:
   - `training/vehicle-classifier/vehicle_classifier_training.ipynb`
   - `training/person-attributes/person_attributes_training.ipynb`

### Step 3: Enable GPU ⚠️ IMPORTANT
```
Runtime → Change runtime type → GPU (T4)
```
**Without GPU, training will be extremely slow!**

### Step 4: Run All Cells
```
Runtime → Run all
```

### Step 5: Wait (~30-45 min)
Training progress displays in real-time:
```
Epoch 1/30 - Train Acc: 0.65 - Val Acc: 0.72
...
Epoch 30/30 - Train Acc: 0.95 - Val Acc: 0.89
```

### Step 6: Download Models
Notebook automatically downloads:
- `*_classifier.onnx` ← Use this for RKNN conversion
- `*_classifier.pth` ← PyTorch weights (backup)
- `config.json` ← Class mappings

---

## 📊 GPU Details (Free Tier)

| GPU | VRAM | Session Limit | Training Time |
|-----|------|---------------|---------------|
| **NVIDIA T4** | 16GB | 12 hours | ~45 min |

You get plenty of buffer with the 12-hour limit!

---

## 📁 Dataset Format

### For Vehicle Classifier:
```
data/
  ├── train/
  │   ├── sedan/
  │   │   ├── img_001.jpg
  │   │   └── ...
  │   ├── SUV/
  │   ├── truck/
  │   ├── van/
  │   ├── pickup/
  │   ├── bus/
  │   ├── motorcycle/
  │   └── bicycle/
  └── val/
      └── (same structure)
```

### Upload Options:

**Option A: Direct Upload (small datasets)**
1. Click folder icon on left
2. Upload to `/content/data/train/`

**Option B: Google Drive (large datasets)**
```python
from google.colab import drive
drive.mount('/content/drive')
# Use /content/drive/MyDrive/your-dataset/
```

**Option C: Download in Notebook**
```python
# Stanford Cars
!wget http://ai.stanford.edu/~jkrause/cars/car_dataset.tgz
# Extract and organize...
```

---

## 💡 Tips

### Maximize Performance
- Use **FP16** if INT8 quantization fails
- Batch size 32 works well for T4 (16GB VRAM)
- 30 epochs = good accuracy/speed tradeoff

### Avoid Disconnection
- Don't close browser tab
- Colab may disconnect after ~90 min of inactivity
- Save checkpoints to Google Drive

### If Training Fails
1. Check GPU is enabled: `torch.cuda.is_available()`
2. Verify dataset structure matches expected format
3. Reduce `BATCH_SIZE` if CUDA OOM error

---

## ✅ After Colab

### On your local machine (rock0 or Mac):

```bash
cd /home/camera-brain/training

# Convert vehicle classifier to RKNN
python rknn_conversion.py \
    --onnx ~/Downloads/vehicle_classifier.onnx \
    --output /home/camera-brain/models/vehicle_classifier.rknn \
    --fp16

# Convert person attributes
python rknn_conversion.py \
    --onnx ~/Downloads/person_attr.onnx \
    --output /home/camera-brain/models/person_attr.rknn \
    --fp16

# Verify models exist
ls -lh /home/camera-brain/models/*.rknn
```

---

## 🆘 Common Issues

| Issue | Solution |
|-------|----------|
| "No GPU available" | Runtime → Change runtime type → GPU |
| "CUDA out of memory" | Reduce BATCH_SIZE to 16 |
| "Session disconnected" | Reconnect, load from Drive checkpoint |
| "Dataset not found" | Check `/content/data/` path |

---

## Next Steps

1. ✅ Complete Colab training
2. ✅ Download ONNX models
3. ✅ Convert to RKNN on rock0
4. 🔄 Integrate with worker code (pending)
5. 🔄 Deploy to cluster

For integration details, see: `docs/VEHICLE_GENDER_DETECTION.md`
