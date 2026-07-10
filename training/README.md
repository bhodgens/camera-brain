# Rock Cluster - Model Training Guide

## Overview

This directory contains training scripts for extending camera-brain detection capabilities:

| Model | Purpose | Input | Output |
|-------|---------|-------|--------|
| **Vehicle Classifier** | Classify vehicle type | 224x224 crop | sedan/SUV/truck/van/pickup/bus/motorcycle/bicycle |
| **Person Attributes** | Gender + Age estimation | 224x112 crop | gender (M/F), age_range (child/teen/adult/senior) |

---

## Quick Start with Google Colab (Free GPU)

### Step 1: Choose Your Model

#### Vehicle Type Classifier
1. Open: [`vehicle-classifier/vehicle_classifier_training.ipynb`](vehicle-classifier/vehicle_classifier_training.ipynb)
2. In Colab: **File** → **Upload Notebook** → Select the `.ipynb` file
3. **Runtime** → **Change runtime type** → **GPU** (T4)
4. **Runtime** → **Run all**

#### Person Attributes
1. Open: [`person-attributes/person_attributes_training.ipynb`](person-attributes/person_attributes_training.ipynb)
2. In Colab: **File** → **Upload Notebook** → Select the `.ipynb` file
3. **Runtime** → **Change runtime type** → **GPU** (T4)
4. **Runtime** → **Run all**

### Step 2: Prepare Dataset

**Option A: Upload Directly to Colab**
```
1. Click folder icon on left panel
2. Upload images to /content/data/train/{class_name}/
3. Structure:
   /content/data/
     ├── train/
     │   ├── sedan/img_001.jpg
     │   ├── SUV/img_002.jpg
     │   └── ...
     └── val/
         ├── sedan/img_101.jpg
         └── ...
```

**Option B: Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

# Then update dataset path in notebook:
train_dataset = VehicleDataset('/content/drive/MyDrive/rock-cluster/data/train', ...)
```

**Option C: Download Dataset in Notebook**
```python
# Stanford Cars (requires approval)
!wget http://ai.stanford.edu/~jkrause/cars/car_dataset.tgz

# Or PA100K for person attributes
!kaggle datasets download -d pa100k-dataset
```

### Step 3: Train (~30-45 minutes)

The notebook will:
1. Load dataset
2. Create MobileNetV3-Small (vehicle) or ResNet18 (person)
3. Train for 30 epochs
4. Show accuracy curves
5. Export to ONNX format

Expected output:
```
Epoch 30/30 - Train Acc: 0.95 - Val Acc: 0.89
✓ Best model saved! Val Acc: 0.89
✓ ONNX model exported: vehicle_classifier.onnx
```

### Step 4: Download Models

After training, notebook automatically downloads:
- `vehicle_classifier_best.pth` - PyTorch weights
- `vehicle_classifier.onnx` - ONNX model (ready for conversion)
- `vehicle_classifier_config.json` - Class mappings
- `training_curves.png` - Training visualization

---

## RKNN Conversion (Local, on rock0 or Mac)

After downloading ONNX from Colab:

### On rock0 (recommended - native ARM64)

```bash
cd /home/camera-brain/training

# Install rknn-toolkit2
pip install rknn-toolkit2

# Convert to RKNN (FP16 mode - no calibration needed)
python rknn_conversion.py \
    --onnx vehicle_classifier.onnx \
    --output /home/camera-brain/models/vehicle_classifier.rknn \
    --fp16

# Or with INT8 quantization (requires calibration images)
python rknn_conversion.py \
    --onnx vehicle_classifier.onnx \
    --output /home/camera-brain/models/vehicle_classifier.rknn \
    --dataset /home/camera-brain/crops/  # Uses existing crops for calibration
```

### On Mac (x86_64)

```bash
# Create virtual environment
python3 -m venv rknn-env
source rknn-env/bin/activate

# Install rknn-toolkit2 (may need Rosetta 2 for M-series Macs)
pip install rknn-toolkit2

# Convert
python rknn_conversion.py \
    --onnx ~/Downloads/vehicle_classifier.onnx \
    --output vehicle_classifier.rknn \
    --fp16

# Copy to rock0
scp vehicle_classifier.rknn caimlas@rock0:/home/camera-brain/models/
```

---

## Dataset Recommendations

### Vehicle Type Classification

| Dataset | Images | Classes | URL |
|---------|--------|---------|-----|
| **Stanford Cars** | 16,185 | 196 car models | http://ai.stanford.edu/~jkrause/cars/car_dataset.html |
| **CompCars** | 136,727 | 1,716 models | http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html |
| **BoxCars116k** | 116,000 | Various | https://github.com/JakubSochor/Boxes |

**Class Mapping (for 8-class output):**
- sedan: compact, mid-size, luxury sedans
- SUV: small/large SUVs, crossovers
- truck: semi trucks, delivery trucks
- van: cargo vans, passenger vans
- pickup: pickup trucks
- bus: city buses, school buses, coaches
- motorcycle: motorcycles, scooters
- bicycle: bicycles, e-bikes

### Person Attributes

| Dataset | Images | Attributes | URL |
|---------|--------|------------|-----|
| **PETA** | 19,000 | gender, age, clothing | http://www.robothought.cn/dataset/PETA.html |
| **PA100K** | 100,000 | 35 attributes | https://github.com/huanghoujing/PAA |
| **Market1501** | 32,668 | gender, age (via annotations) | https://www.aitrial.com/dataset |

**Label Format:**
- gender: 0=Male, 1=Female
- age_range: 0=Child (0-12), 1=Teen (13-19), 2=Adult (20-59), 3=Senior (60+)

---

## Testing Trained Models

### Test Vehicle Classifier

```python
# On rock0, test with existing crops
python test_vehicle_classifier.py \
    --model /home/camera-brain/models/vehicle_classifier.rknn \
    --crop /home/camera-brain/crops/20260710_123456_cam1_car_0.85.jpg

# Expected output:
# Prediction: SUV (confidence: 0.92)
```

### Integration with Worker

After RKNN conversion:

1. **Copy model to workers:**
   ```bash
   for node in rock2 rock3 rock4 rock5; do
     scp /home/camera-brain/models/vehicle_classifier.rknn rock@$node:/home/camera-brain/models/
   done
   ```

2. **Update worker code** to load and run classifier (pending implementation)

3. **Test with live camera feed:**
   ```bash
   ssh rock@rock2
   journalctl -fu camera-brain-worker
   # Should see: "Vehicle type: SUV" in detection logs
   ```

---

## Troubleshooting

### Colab Issues

**Problem:** "No GPU available"
- **Solution:** Runtime → Change runtime type → GPU (may need to wait during high demand)

**Problem:** "Session disconnected"
- **Solution:** Colab free tier has 12-hour limit. Save checkpoints regularly:
  ```python
  torch.save(model.state_dict(), '/content/drive/MyDrive/checkpoint_epoch15.pth')
  ```

**Problem:** "CUDA out of memory"
- **Solution:** Reduce batch size in notebook:
  ```python
  BATCH_SIZE = 16  # was 32
  ```

### RKNN Conversion Issues

**Problem:** "rknn_init failed"
- **Solution:** Ensure driver is loaded: `lsmod | grep rknpu`

**Problem:** "Model loading failed"
- **Solution:** Check ONNX validity:
  ```bash
  python -c "import onnx; onnx.load_model('model.onnx')"
  ```

---

## Next Steps

After successful training and conversion:

1. ✅ Models ready in `/home/camera-brain/models/`
2. 🔄 Update worker code to use new classifiers
3. 🔄 Update database schema for new fields
4. 🔄 Update chat/app.py schema context
5. 🔄 Deploy to cluster

See `docs/VEHICLE_GENDER_DETECTION.md` for integration details.
