#!/usr/bin/env python3
"""
Test script to verify all required images are available and can be loaded
"""

import cv2
import os

def test_images():
    print("🔍 Testing image loading for Dr. Strange Shield System...")
    print("="*60)

    current_directory = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")

    image_files = {
        'Welcome/Example': 'images/example.png',
        'KEY 1 Gesture': 'images/position_1.png',
        'KEY 2 Gesture': 'images/position_2.png',
        'KEY 3 Gesture': 'images/position_3.png',
        'KEY 4 Gesture': 'images/position_4.png'
    }

    all_good = True

    for name, path in image_files.items():
        full_path = os.path.join(current_directory, path)
        print(f"📁 Checking {name}: {path}")

        if os.path.exists(full_path):
            try:
                img = cv2.imread(full_path)
                if img is not None:
                    height, width = img.shape[:2]
                    print(f"   ✅ Loaded successfully - Size: {width}x{height}")
                else:
                    print(f"   ❌ File exists but couldn't load image")
                    all_good = False
            except Exception as e:
                print(f"   ❌ Error loading image: {e}")
                all_good = False
        else:
            print(f"   ❌ File not found!")
            all_good = False
        print()

    print("="*60)
    if all_good:
        print("🎉 All images loaded successfully!")
        print("✅ Enhanced UI will work properly with visual instructions")
    else:
        print("⚠️  Some images are missing or couldn't be loaded")
        print("💡 The enhanced UI will still work but without some visual guides")

    print("="*60)
    return all_good

if __name__ == "__main__":
    test_images()
