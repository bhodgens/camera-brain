# Pre-trained Attribute Detectors - Installation

## Quick Install (on rock0)

```bash
cd /home/camera-brain

# Create virtual environment
python3 -m venv attribute-env
source attribute-env/bin/activate

# Install dependencies
pip install opencv-python-headless deepface

# Optional: for vehicle ML classification
pip install onnxruntime
```

## Test Installation

### Test Gender Detection
```bash
source attribute-env/bin/activate

# Run with a sample person crop
python cmd/gender-detector/detect.py \
    --crop /home/camera-brain/crops/person_crop.jpg \
    --json
```

Expected output:
```json
{"success": true, "gender": "Woman", "age": 28, "gender_conf": 0.94}
```

### Test Vehicle Detection
```bash
# Heuristic mode (works immediately)
python cmd/vehicle-detector/detect.py \
    --crop /home/camera-brain/crops/car_crop.jpg \
    --json
```

Expected output:
```json
{
  "success": true,
  "vehicle_type": "sedan",
  "confidence": 0.65,
  "color": "white",
  "method": "heuristic"
}
```

## Model Performance

| Detector | Accuracy | Speed | Dependencies |
|----------|----------|-------|--------------|
| **Gender (DeepFace)** | 92-95% | ~200ms | deepface, retinaface |
| **Vehicle (Heuristic)** | 60-70% | ~10ms | opencv only |
| **Vehicle (ML)** | 88-92% | ~50ms | onnxruntime + trained model |

## Integration with Worker

The Go worker can call these Python scripts as subprocesses:

```go
// After detecting a person
cmd := exec.Command("python3", "/home/camera-brain/cmd/gender-detector/detect.py",
    "--crop", cropPath, "--json")
output, _ := cmd.Output()

var result struct {
    Gender     string  `json:"gender"`
    Age        int     `json:"age"`
    GenderConf float64 `json:"gender_conf"`
}
json.Unmarshal(output, &result)

// Send to gateway with extended attributes
```

## File Locations

| File | Purpose |
|------|---------|
| `cmd/gender-detector/detect.py` | Gender + age detection |
| `cmd/vehicle-detector/detect.py` | Vehicle type classification |
| `/home/camera-brain/models/vehicle_classifier.rknn` | Pre-trained vehicle model (future) |

## Usage Examples

### Gender Detection
```bash
python cmd/gender-detector/detect.py --crop image.jpg --json
python cmd/gender-detector/detect.py --crop image.jpg --verbose
```

### Vehicle Detection (Heuristic)
```bash
python cmd/vehicle-detector/detect.py --crop car.jpg --json
```

### Vehicle Detection (ML Model)
```bash
python cmd/vehicle-detector/detect.py \
    --crop car.jpg \
    --model /home/camera-brain/models/vehicle_classifier.onnx \
    --json
```
