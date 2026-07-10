# Extended Detection Capabilities Plan

## Current State

The camera-brain system uses YOLOv5 INT8 quantized model running on RK3568 NPU:
- **80 COCO classes**: person, car, truck, bicycle, etc.
- **Output**: class_name, confidence, bbox [x1, y1, x2, y2]
- **Throughput**: ~2 fps per camera at 640x640

**Limitations:**
- No color information
- No vehicle attributes (make, model, year)
- No license plate recognition
- No speed/velocity tracking
- No person attributes (clothing, age range, gender)

---

## Extended Attributes Architecture

### 1. Color Detection

**Approach**: Dominant color extraction from crop regions

```
Detection Crop → Resize (224x224) → Color Space Analysis → Dominant Color
                     │
                     └──→ CNN Classifier → Color Label
```

**Implementation Options:**

#### A. Simple Color Histogram (CPU, ~5ms per crop)
```python
import cv2
import numpy as np

def get_dominant_color(crop):
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Define color ranges
    color_ranges = {
        'red': ([0, 50, 50], [10, 255, 255]),
        'blue': ([100, 50, 50], [130, 255, 255]),
        'green': ([40, 50, 50], [70, 255, 255]),
        # ... etc
    }

    dominant = 'unknown'
    max_pixels = 0

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        count = np.sum(mask > 0)
        if count > max_pixels:
            max_pixels = count
            dominant = color

    return dominant
```

**Pros:** Fast, no ML needed, works on edge
**Cons:** Lighting sensitive, limited to predefined colors

#### B. CNN Color Classifier (NPU, ~20ms per crop)
Train a lightweight ResNet18 on color-labeled vehicle dataset.

**Dataset:** Stanford Cars (color-annotated), BoxCars116k
**Model:** ResNet18 → 10 color classes (white, black, red, blue, silver, gray, green, yellow, brown, orange)
**Deployment:** Convert to RKNN, run on NPU

---

### 2. Speed/Velocity Estimation

**Approach**: Multi-frame tracking with camera calibration

```
Frame T:   Detection (x1, y1, x2, y2) ─┐
                                       ├──→ Track ID assignment ─→ Position over time
Frame T+1: Detection (x1', y1', x2', y2') ─┘       │
                                                   ↓
                                    Camera calibration + Real-world scale
                                                   │
                                                   ↓
                                    Pixels/frame → Meters/second
```

**Requirements:**
1. **Object Tracking**: DeepSORT, ByteTrack, or BoT-SORT
2. **Camera Calibration**: Homography matrix (image plane → ground plane)
3. **Frame Timing**: Accurate timestamp per detection
4. **Reference Scale**: Known object size or manual calibration

**Implementation:**
```python
class SpeedEstimator:
    def __init__(self, camera_matrix, homography):
        self.tracks = {}  # track_id -> [(timestamp, position), ...]
        self.homography = homography  # image → ground plane

    def update(self, detections, timestamp):
        """
        detections: [(bbox, track_id), ...]
        """
        for bbox, track_id in detections:
            center = self._get_ground_point(bbox)

            if track_id not in self.tracks:
                self.tracks[track_id] = []

            self.tracks[track_id].append((timestamp, center))

            # Keep last 2 seconds of positions
            cutoff = timestamp - 2.0
            self.tracks[track_id] = [
                (t, p) for t, p in self.tracks[track_id] if t >= cutoff
            ]

            # Calculate speed
            if len(self.tracks[track_id]) >= 2:
                speed = self._calculate_speed(track_id)
                return speed

    def _get_ground_point(self, bbox):
        """Convert bbox bottom-center to ground plane coordinates"""
        x1, y1, x2, y2 = bbox
        bottom_center = [(x1 + x2) / 2, y2]
        ground_point = cv2.perspectiveTransform(
            np.array([[bottom_center]], dtype=np.float32),
            self.homography
        )
        return ground_point[0][0]

    def _calculate_speed(self, track_id):
        """Calculate speed in m/s from track history"""
        history = self.tracks[track_id]
        if len(history) < 2:
            return None

        first_t, first_pos = history[0]
        last_t, last_pos = history[-1]

        time_delta = last_t - first_t  # seconds
        distance = np.linalg.norm(last_pos - first_pos)  # meters

        return distance / time_delta  # m/s
```

**Accuracy factors:**
- Camera angle (top-down = more accurate)
- Frame rate (higher = better speed resolution)
- Calibration quality
- Object occlusion handling

---

### 3. Vehicle Make/Model Classification

**Approach**: Two-stage classification

```
Crop → Vehicle Type (car/truck/bus) ─┐
                                     ├──→ Specific Model
Make Classifier ──────────────────────┘
```

**Models:**
1. **Vehicle Type**: Lightweight MobileNetV3 (5 classes: car, truck, bus, van, motorcycle)
2. **Make**: Brand classifier (Toyota, Ford, BMW, etc.)
3. **Model**: Year-specific model classifier

**Datasets:**
- Stanford Cars (196 car models, 16,185 images)
- CompCars (1,716 car models, 136,727 images)
- BoxCars116k (116,000+ vehicle images with bounding boxes)

**Deployment Strategy:**
```
┌─────────────────────────────────────────────┐
│ Worker Node (RK3568)                        │
│  ├─ YOLOv5 detection (NPU)                  │
│  ├─ Color extraction (CPU)                  │
│  └─ Send crop to gateway                    │
│                                             │
│ Gateway (rock0 - more CPU)                  │
│  ├─ Receive crop                            │
│  ├─ Run make/model classifier (CPU/GPU)     │
│  └─ Store: class, color, make, model        │
└─────────────────────────────────────────────┘
```

---

### 4. License Plate Recognition (LPR/ALPR)

**Pipeline:**
```
Vehicle Crop → Plate Detector → Plate Crop → OCR → Text Post-processing
                 (YOLO)           (300x80)   (CRNN)     (regex)
```

**Components:**

1. **Plate Detection**: YOLOv5-small trained on license plates
   - Input: 640x640 vehicle region
   - Output: Plate bounding box

2. **Plate OCR**: CRNN or LPRNet
   - Input: 300x80 plate crop (normalized)
   - Output: Alphanumeric text

**Pre-trained Options:**
- **PaddleOCR**: Multi-language, supports ALPR
- **EasyOCR**: Python-based, 80+ languages
- **OpenALPR**: Commercial, high accuracy
- **LPRNet**: Lightweight, real-time capable

**RK3568 Performance Estimate:**
- Plate detection: ~50ms (NPU)
- OCR: ~100ms (CPU, optimized)
- **Total: ~150ms per vehicle**

---

### 5. Person Attributes

**Attributes to extract:**
- Gender (male/female)
- Age range (child/teen/adult/senior)
- Clothing color (upper/lower body)
- Carrying objects (backpack, bag, umbrella)

**Model Architecture:**
```
Person Crop (256x128) → Shared Backbone (ResNet18)
                          ├─ Gender Head (2-class)
                          ├─ Age Head (4-class)
                          ├─ Upper Color (10-class)
                          ├─ Lower Color (10-class)
                          └─ Accessories (multi-label)
```

**Datasets:**
- Market1501 (person ReID, some attributes)
- DukeMTMC-ReID
- PETA (Person Attributes dataset)
- PA100K (Large-scale Person Attribute)

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)
1. **Color Detection** (CPU histogram)
   - No model training needed
   - Immediate deployment
   - Works for vehicles and large objects

2. **Basic Tracking**
   - Simple centroid tracking
   - Frame-to-frame association
   - Speed estimation (relative, not calibrated)

### Phase 2: Medium Complexity (1 week)
3. **Vehicle Make/Model**
   - Deploy pre-trained model on gateway
   - Workers send crops for vehicles only
   - Store make, model in observations table

4. **License Plate Detection**
   - Add plate detector to worker
   - Plate crops sent to gateway for OCR
   - Store plate text in database

### Phase 3: Advanced Features (2-3 weeks)
5. **Person Attributes**
   - Train multi-task attribute model
   - Deploy to workers with NPU
   - Privacy considerations (GDPR compliance)

6. **Calibrated Speed Estimation**
   - Manual camera calibration UI
   - Homography estimation
   - Real-world speed in km/h or mph

---

## Database Schema Updates

```sql
-- Add columns to observations table
ALTER TABLE observations ADD COLUMN color VARCHAR(20);
ALTER TABLE observations ADD COLUMN vehicle_make VARCHAR(30);
ALTER TABLE observations ADD COLUMN vehicle_model VARCHAR(30);
ALTER TABLE observations ADD COLUMN license_plate VARCHAR(20);
ALTER TABLE observations ADD COLUMN speed_mps FLOAT;  -- meters per second
ALTER TABLE observations ADD COLUMN tracking_id UUID;
ALTER TABLE observations ADD COLUMN person_gender VARCHAR(10);
ALTER TABLE observations ADD COLUMN person_age_range VARCHAR(20);
ALTER TABLE observations ADD COLUMN accessories TEXT[];  --JSON array

-- Index for common queries
CREATE INDEX idx_color ON observations(color);
CREATE INDEX idx_vehicle_make ON observations(vehicle_make);
CREATE INDEX idx_license_plate ON observations(license_plate);
CREATE INDEX idx_tracking_id ON observations(tracking_id);
```

---

## Gateway API Extensions

### POST /observations (updated)

```json
{
  "camera_id": "uuid",
  "worker_id": "rock1",
  "detected_at": "2026-07-10T12:34:56Z",
  "class_name": "car",
  "confidence": 0.89,
  "bbox": [x1, y1, x2, y2],
  "crop_path": "/path/to/crop.jpg",
  "crop_data": "base64_encoded_image",  // NEW: for gateway processing
  "attributes": {                        // NEW: extensible attributes
    "color": "red",
    "vehicle_make": "toyota",
    "vehicle_model": "camry",
    "license_plate": "ABC-1234",
    "speed_mps": 15.5
  }
}
```

---

## Performance Impact

| Feature | CPU Usage | NPU Usage | Memory | Latency Added |
|---------|-----------|-----------|--------|---------------|
| Color (histogram) | +5% | 0% | negligible | +2ms |
| Color (CNN) | 0% | +10% | 25MB | +20ms |
| Speed (tracking) | +10% | 0% | 50MB | +5ms |
| Make/Model | +30% (gateway) | 0% | 100MB | +50ms |
| License Plate | +20% | +15% | 75MB | +150ms |
| Person Attributes | +15% | +20% | 50MB | +40ms |

**Recommendation**: Start with color histogram (Phase 1), then add features incrementally based on performance monitoring.

---

## Files to Create/Modify

### New Files
- `/home/camera-brain/worker-src/color.go` - Color extraction
- `/home/camera-brain/worker-src/tracker.go` - Multi-object tracking
- `/home/camera-brain/gateway/attributes.go` - Make/model classifier
- `/home/camera-brain/gateway/ocr.go` - License plate OCR

### Modified Files
- `/home/camera-brain/worker-src/crop.go` - Send crop data + attributes
- `/home/camera-brain/gateway/handlers.go` - Updated observations handler
- `/home/camera-brain/storage/schema.sql` - New columns
- `chat/app.py` - Updated schema context for new fields

---

## Next Steps

1. **Deploy chat UI fix** (user messages visible)
2. **Choose Phase 1 features** to implement
3. **Set up model training pipeline** for vehicle attributes
4. **Test on single worker** before cluster-wide deployment
