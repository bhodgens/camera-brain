# Gender/Age Detector (Pre-trained)

Uses DeepFace library with pre-trained models - no training required!

## Quick Install

```bash
pip install deepface opencv-python
```

## Usage

```python
from deepface import DeepFace

# Analyze a face crop
result = DeepFace.analyze(
    img_path="person_crop.jpg",
    actions=['gender', 'age'],
    enforce_detection=False,  # Don't fail if no face detected
    silent=True
)

print(f"Gender: {result['dominant_gender']} ({result['gender']['Woman']:.1f}%)")
print(f"Age: {result['age']}")
```

## Integration with Camera Brain

The `crop.go` worker code can call this Python script as a subprocess for person detections:

```bash
python detect_gender.py --crop /path/to/person_crop.jpg
```

Output (JSON):
```json
{"gender": "Woman", "age": 28, "gender_conf": 0.94}
```
