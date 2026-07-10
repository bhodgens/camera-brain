# Vehicle Type + Gender Detection Implementation

## Overview

This document details implementation of:
1. **Vehicle Type Classification**: 7-class classifier (car, truck, bus, van, SUV, motorcycle, bicycle)
2. **Gender Classification**: Binary classifier (male/female) for person detections

---

## 1. Vehicle Type Classification

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ Option A: Secondary Classifier (Recommended for quick deploy)│
├──────────────────────────────────────────────────────────────┤
│ YOLOv5 detects "car" (class 2) ─┬─> Crop resize (224x224)   │
│                                 │                           │
│                                 └─> MobileNetV3 → 7 classes │
│                                                             │
│ Advantage: No retraining of YOLOv5 needed                   │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ Option B: Extended YOLOv5                                     │
├──────────────────────────────────────────────────────────────┤
│ Retrain YOLOv5 with 7 additional vehicle type classes:       │
│ - Current: "car" (single class)                              │
│ - Extended: sedan, SUV, truck, van, bus, pickup, motorcycle  │
│                                                               │
│ Advantage: Single model, better consistency                   │
│ Disadvantage: Requires model retraining                       │
└──────────────────────────────────────────────────────────────┘
```

### Recommended: Option A (Secondary Classifier)

**Why:**
- No YOLOv5 retraining needed
- Can use pre-trained vehicle classifier
- Deploys incrementally (only on gateway initially)
- Lower risk, faster deployment

### Vehicle Type Classes

| Class ID | Type | Example |
|----------|------|---------|
| 0 | sedan | Toyota Camry, Honda Civic |
| 1 | SUV | Ford Explorer, BMW X5 |
| 2 | truck | Semi truck, delivery truck |
| 3 | van | Cargo van, passenger van |
| 4 | pickup | Ford F-150, Chevy Silverado |
| 5 | bus | City bus, school bus |
| 6 | motorcycle | Motorcycle, scooter |
| 7 | bicycle | Bicycle, e-bike |

### Model: MobileNetV3-Small

```python
# PyTorch model definition
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

def create_vehicle_classifier(num_classes=8):
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

# Export to ONNX, then to RKNN for NPU deployment
```

**Model Specs:**
- Parameters: 2.5M
- Size: ~10MB (FP32), ~3MB (INT8)
- NPU inference: ~15ms on RK3568

### Dataset Options

1. **Stanford Cars** - 196 car models, ~16k images
2. **CompCars** - 1,716 models, surveillance + web views
3. **BoxCars116k** - 116k vehicle images with bounding boxes
4. **VeRi-776** - 50k+ images, 776 vehicle IDs (good for type classification)

**Training Steps:**
```bash
# 1. Prepare dataset (convert to vehicle type labels)
# 2. Train MobileNetV3
python train_vehicle.py --data /path/to/combined --classes 8 --epochs 50

# 3. Export to ONNX
torch.onnx.export(model, dummy_input, "vehicle_classifier.onnx")

# 4. Convert to RKNN
python convert_to_rknn.py --input vehicle_classifier.onnx --output vehicle_classifier.rknn
```

---

## 2. Gender Classification

### Architecture

```
Person Crop Detection
        │
        ▼
┌───────────────────────────────────────────────┐
│  ResNet18 Multi-Task Attribute Network        │
├───────────────────────────────────────────────┤
│  Input: 128x256 person crop (full body)       │
│                                               │
│  Backbone: ResNet18 (shared features)         │
│    ├─ Global Pool                             │
│    │                                          │
│    ├─ Gender Head: FC(512) → FC(128) → FC(1) │
│    │   Output: sigmoid → [0=male, 1=female]  │
│    │                                          │
│    ├─ Age Head: FC(512) → FC(128) → FC(4)    │
│    │   Output: softmax → [child, teen,       │
│    │                      adult, senior]     │
│    │                                          │
│    └─ Clothing Head (optional)                │
│        Output: multi-label (8 categories)     │
└───────────────────────────────────────────────┘
```

### Model Specs

| Component | Input Size | Parameters | Inference Time |
|-----------|------------|------------|----------------|
| Backbone | 128x256 | 11M (ResNet18) | ~25ms |
| Gender Head | - | 65k | ~2ms |
| Age Head | - | 2k | ~1ms |
| **Total** | - | **~11M** | **~30ms** |

### Datasets

| Dataset | Images | Attributes | Notes |
|---------|--------|------------|-------|
| **PETA** | 19k | gender, age, clothing | Multi-camera, person ReID |
| **PA100K** | 100k | 35 attributes | Large-scale, diverse |
| **Market1501** | 32k | gender, age labels available | Person ReID benchmark |
| **DukeMTMC-ReID** | 36k | gender annotations | Multi-view, challenging |

### Training Code (PyTorch)

```python
import torch
import torch.nn as nn
import torchvision.models as models

class PersonAttributeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

        # Gender head (binary)
        self.gender_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        # Age head (4 classes)
        self.age_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        features = self.features(x).view(-1, self.feature_dim)
        gender_logits = self.gender_head(features)
        age_logits = self.age_head(features)
        return gender_logits, age_logits

    def predict_gender(self, x):
        gender_logits, _ = self.forward(x)
        prob = torch.sigmoid(gender_logits)
        return (prob > 0.5).long(), prob
```

### Training Command

```bash
python train_person_attr.py \
    --data /path/to/PETA \
    --batch_size 64 \
    --epochs 30 \
    --lr 0.001 \
    --output person_attr_net.onnx
```

### Accuracy Considerations

| Condition | Gender Accuracy | Age Accuracy |
|-----------|-----------------|--------------|
| Full body, clear view | 92-95% | 80-85% |
| Partial occlusion | 75-85% | 60-70% |
| Low light | 65-75% | 50-60% |
| Distant/small crop | 60-70% | 45-55% |

**Important Notes:**
- Gender is binary in training data ( limitations in available datasets)
- Accuracy drops significantly for non-frontal views
- Age estimation has higher variance than gender
- **Must include confidence scores in output**
- Consider local privacy regulations (GDPR, BIPA, etc.)

---

## Integration Pipeline

### Worker-Side Processing

```go
// bin/npu.go - after YOLOv5 detection
type Detection struct {
    ClassID    int
    ClassName  string
    Confidence float32
    BBox       [4]int
    Crop       []byte  // JPEG-encoded crop

    // NEW: Extended attributes
    VehicleType string `json:"vehicle_type,omitempty"`  // car/sedan, car/SUV, etc.
    Gender      string `json:"gender,omitempty"`        // male, female
    GenderConf  float32 `json:"gender_conf,omitempty"`
}

func annotateWithAttributes(det *Detection) error {
    if det.ClassName == "car" {
        // Run vehicle type classifier
        typ, err := classifyVehicle(det.Crop)
        if err == nil {
            det.VehicleType = typ
        }
    } else if det.ClassName == "person" {
        // Run gender classifier
        gender, conf, err := classifyPerson(det.Crop)
        if err == nil {
            det.Gender = gender
            det.GenderConf = conf
        }
    }
    return nil
}
```

### Gateway Observation Handler

```go
// cmd/gateway/main.go - observationsHandler
type ObservationInsert struct {
    CameraID      uuid.UUID
    DetectedAt    time.Time
    Type          string
    ClassName     string
    Confidence    float32
    BBox          []int
    CropPath      string
    Color         *string      `json:"color"`
    VehicleType   *string      `json:"vehicle_type"`
    Gender        *string      `json:"gender"`
    GenderConf    *float32     `json:"gender_conf"`
    // ... existing fields
}

func (s *Server) handleObservations(w http.ResponseWriter, r *http.Request) {
    var obs ObservationInsert
    if err := json.NewDecoder(r.Body).Decode(&obs); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    // Insert into database with extended attributes
    _, err := s.db.Exec(`
        INSERT INTO observations (
            camera_id, detected_at, type, class_name, confidence, bbox,
            crop_path, color, vehicle_type, gender, gender_conf
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
    `,
        obs.CameraID, obs.DetectedAt, obs.Type, obs.ClassName,
        obs.Confidence, obs.BBox, obs.CropPath,
        obs.Color, obs.VehicleType, obs.Gender, obs.GenderConf,
    )
    // ...
}
```

### Database Schema

```sql
-- Add new columns
ALTER TABLE observations ADD COLUMN color VARCHAR(20);
ALTER TABLE observations ADD COLUMN vehicle_type VARCHAR(20);
ALTER TABLE observations ADD COLUMN gender VARCHAR(10);
ALTER TABLE observations ADD COLUMN gender_conf REAL;

-- Indexes for common queries
CREATE INDEX idx_vehicle_type ON observations(vehicle_type) WHERE vehicle_type IS NOT NULL;
CREATE INDEX idx_gender ON observations(gender) WHERE gender IS NOT NULL;

-- View for easy querying
CREATE VIEW observations_enhanced AS
SELECT
    o.*,
    c.name as camera_name,
    c.location as camera_location
FROM observations o
LEFT JOIN cameras c ON o.camera_id = c.id;
```

### Updated Chat Schema Context

```python
# chat/app.py - SCHEMA_CONTEXT update
SCHEMA_CONTEXT = """
Database schema:
- observations(id, camera_id, detected_at, type, class_name, confidence, bbox, crop_path,
              color, vehicle_type, gender, gender_conf)
- cameras(id, name, rtsp_url, location, active, created_at)

Available fields: id, camera_id, detected_at, type, class_name, confidence, bbox, crop_path,
                  color, vehicle_type, gender, gender_conf, name, rtsp_url, location, active, created_at

Available class_name values: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat...
Available vehicle_type values: sedan, SUV, truck, van, pickup, bus, motorcycle, bicycle
Available gender values: male, female

NOT available: color (for detections before 2026-07-10), make, model (separate from vehicle_type),
               license_plate, speed, direction, emotion, clothing details, age (only gender), etc.
"""
```

---

## Deployment Steps

### Phase 1: Model Training (Local/Cloud)

1. **Vehicle Classifier**
   ```bash
   # On training machine with GPU
   cd /home/camera-brain/training/vehicle-classifier
   python train.py --data combined_dataset --classes 8 --epochs 50
   python export_onnx.py --checkpoint best.pth --output vehicle_classifier.onnx
   python rknn_convert.py --onnx vehicle_classifier.onnx --output vehicle_classifier.rknn
   scp vehicle_classifier.rknn rock0:/home/camera-brain/models/
   ```

2. **Person Attribute Model**
   ```bash
   cd /home/camera-brain/training/person-attributes
   python train.py --data PETA --epochs 30
   python export_onnx.py --multi-task
   python rknn_convert.py --onnx person_attr.onnx --output person_attr.rknn
   scp person_attr.rknn rock0:/home/camera-brain/models/
   ```

### Phase 2: Worker Code Updates

3. **Add classifier wrappers** (`bin/vehicle_classifier.go`, `bin/person_classifier.go`)

4. **Update detection pipeline** (`bin/npu.go`) - call classifiers after YOLOv5

5. **Update crop.go** - send extended attributes to gateway

### Phase 3: Gateway Updates

6. **Update handlers.go** - accept new fields

7. **Update schema.sql** - add columns, indexes

8. **Update chat/app.py** - new schema context

### Phase 4: Deploy & Test

9. **Deploy to single worker** (rock2) - test vehicle + gender detection

10. **Verify database** - check new columns populated

11. **Test chat queries** - "Show me all female detections", "What SUVs were detected?"

12. **Cluster-wide deploy** - copy to rock3-5

---

## Example Queries (After Deployment)

```
User: "Show me all SUVs detected today"
SQL:  SELECT * FROM observations
      WHERE vehicle_type = 'SUV'
      AND DATE(detected_at) = CURRENT_DATE
      LIMIT 50

User: "How many male vs female detections on camera 2?"
SQL:  SELECT gender, COUNT(*) as count
      FROM observations
      WHERE camera_id = (SELECT id FROM cameras WHERE name = 'cam2')
      AND gender IS NOT NULL
      GROUP BY gender

User: "What vehicle types were detected?"
SQL:  SELECT vehicle_type, COUNT(*) as count
      FROM observations
      WHERE vehicle_type IS NOT NULL
      GROUP BY vehicle_type

User: "Show me trucks from the last hour"
SQL:  SELECT * FROM observations
      WHERE vehicle_type IN ('truck', 'bus')
      AND detected_at >= NOW() - INTERVAL '1 hour'
      LIMIT 50
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Privacy concerns (gender) | High | Add opt-out config, document usage, include confidence scores |
| Model accuracy issues | Medium | Start with gateway-only deployment, collect feedback |
| Performance regression | Medium | Profile on single worker before cluster deploy |
| Database migration issues | Low | Test on dev database first, backup before ALTER TABLE |

---

## Success Criteria

- [ ] Vehicle type classifier: >90% accuracy on sedan/SUV/truck
- [ ] Gender classifier: >85% accuracy on clear person crops
- [ ] NPU inference: <50ms combined overhead per frame
- [ ] Database queries: <100ms for new attribute-based queries
- [ ] Chat UI: Natural language queries for vehicle_type and gender work
