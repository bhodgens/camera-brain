# Person Attributes Training - Google Colab

## Quick Start

### Step 1: Open the Colab Notebook

1. Go to https://colab.research.google.com
2. Click **Upload** → **GitHub** tab
3. Paste: `https://github.com/YOUR_USERNAME/rock-cluster/blob/main/training/person-attributes/person_attributes_training.ipynb`
4. **Runtime** → **Change runtime type** → **GPU** (T4)

### Step 2: Dataset Options

**Option A: Use PETA Dataset** (Recommended)
- Download from: http://www.robothought.cn/dataset/PETA.zip
- Requires institutional email for access

**Option B: Use PA100K** (Easier access)
- Available on Kaggle: "PA100K Person Attributes"
- 100k images with 35 attributes including gender

**Option C: Synthetic Demo** (For testing pipeline)
- Notebook includes code to create synthetic data
- Lower accuracy but tests full pipeline

### Step 3: Run All Cells (~30 minutes)

### Step 4: Download Models
- `person_attr_best.pth` - PyTorch weights
- `person_attr.onnx` - ONNX format (ready for RKNN conversion)

---

## Output Classes

| Attribute | Classes |
|-----------|---------|
| **Gender** | Male, Female |
| **Age Range** | Child (0-12), Teen (13-19), Adult (20-59), Senior (60+) |

## Expected Accuracy

| Attribute | Expected Accuracy |
|-----------|-------------------|
| Gender | 92-95% (clear view), 75-85% (body only) |
| Age Range | 80-85% |
