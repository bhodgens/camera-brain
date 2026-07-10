#!/usr/bin/env python3
"""
Pre-trained Gender and Age Detection
Uses DeepFace library - no training required!

Usage:
    python detect.py --crop /path/to/person_crop.jpg
    python detect.py --crop image.jpg --json

Output:
    JSON format for easy integration with Go worker
"""

import argparse
import json
import sys

try:
    from deepface import DeepFace
except ImportError:
    print("ERROR: DeepFace not installed", file=sys.stderr)
    print("Install with: pip install deepface opencv-python", file=sys.stderr)
    sys.exit(1)


def detect_gender_age(image_path: str, silent: bool = True) -> dict:
    """
    Detect gender and age from an image.
    """
    try:
        results = DeepFace.analyze(
            img_path=image_path,
            actions=['gender', 'age'],
            enforce_detection=False,
            detector_backend='retinaface',
            align=True,
            silent=silent
        )

        if isinstance(results, list):
            if len(results) == 0:
                return {
                    'success': False,
                    'error': 'No face detected',
                    'gender': None,
                    'age': None,
                    'gender_conf': None
                }
            result = results[0]
        else:
            result = results

        gender_dict = result.get('gender', {})
        dominant_gender = result.get('dominant_gender', 'Unknown')

        if gender_dict:
            gender_conf = gender_dict.get(dominant_gender, 0) / 100.0
        else:
            gender_conf = 0.0

        age_val = result.get('age', None)
        if hasattr(age_val, 'item'):
            age_val = int(age_val.item())

        return {
            'success': True,
            'gender': dominant_gender.lower() if dominant_gender else None,
            'age': age_val,
            'gender_conf': round(float(gender_conf), 3),
            'all_gender_scores': {k: round(float(v), 2) for k, v in gender_dict.items()}
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'gender': None,
            'age': None,
            'gender_conf': None
        }


def main():
    parser = argparse.ArgumentParser(description='Pre-trained gender/age detection')
    parser.add_argument('--crop', required=True, help='Path to person crop image')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')

    args = parser.parse_args()
    result = detect_gender_age(args.crop, silent=not args.verbose)

    if args.json:
        print(json.dumps(result))
    else:
        if result['success']:
            print(f"Gender: {result['gender']} (confidence: {result['gender_conf']:.1%})")
            print(f"Age: {result['age']}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)


if __name__ == '__main__':
    main()
