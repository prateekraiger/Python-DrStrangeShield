#!/usr/bin/env python3
"""
Installation verification script for Dr. Strange Shield System
Checks if all required dependencies are properly installed
"""

import sys
import importlib
import subprocess

def check_package(package_name, import_name=None, version_attr=None):
    """Check if a package is installed and optionally get its version"""
    if import_name is None:
        import_name = package_name

    try:
        module = importlib.import_module(import_name)

        # Try to get version
        version = "Unknown"
        if version_attr:
            version = getattr(module, version_attr, "Unknown")
        elif hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'version'):
            version = module.version

        print(f"✅ {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: Not installed ({e})")
        return False
    except Exception as e:
        print(f"⚠️  {package_name}: Installed but error getting version ({e})")
        return True

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  Warning: Python 3.8+ recommended")
        return False
    return True

def check_camera():
    """Check if camera is accessible"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Camera: Accessible")
                return True
            else:
                print("⚠️  Camera: Connected but no frame received")
                return False
        else:
            print("❌ Camera: Not accessible (check if camera is connected)")
            return False
    except Exception as e:
        print(f"❌ Camera: Error checking camera ({e})")
        return False

def check_project_files():
    """Check if required project files exist"""
    import os

    required_files = {
        'Main Scripts': ['shield.py', 'shield_enhanced.py', 'launcher.py'],
        'Utilities': ['utils.py'],
        'Models': ['models/model_svm.sav'],
        'Effects': ['effects/shield.mp4'],
        'Images': ['images/example.png', 'images/position_1.png',
                  'images/position_2.png', 'images/position_3.png', 'images/position_4.png']
    }

    all_good = True

    for category, files in required_files.items():
        print(f"\n📁 {category}:")
        for file in files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"   ✅ {file} ({size:,} bytes)")
            else:
                print(f"   ❌ {file} (missing)")
                all_good = False

    return all_good

def main():
    print("🔍 Dr. Strange Shield System - Installation Verification")
    print("="*70)

    # Check Python version
    python_ok = check_python_version()
    print()

    # Check required packages
    print("📦 Checking Required Packages:")
    print("-" * 40)

    packages = [
        ('OpenCV', 'cv2'),
        ('MediaPipe', 'mediapipe'),
        ('NumPy', 'numpy'),
        ('Scikit-learn', 'sklearn'),
        ('PyVirtualCam', 'pyvirtualcam'),
        ('Pickle', 'pickle'),
        ('DateTime', 'datetime'),
        ('OS', 'os'),
        ('Signal', 'signal'),
        ('ArgParse', 'argparse'),
        ('Time', 'time'),
        ('Sys', 'sys')
    ]

    packages_ok = 0
    for package_name, import_name in packages:
        if check_package(package_name, import_name):
            packages_ok += 1

    print(f"\n📊 Package Status: {packages_ok}/{len(packages)} packages available")

    # Check camera
    print("\n📹 Checking Camera:")
    print("-" * 20)
    camera_ok = check_camera()

    # Check project files
    print("\n📂 Checking Project Files:")
    print("-" * 30)
    files_ok = check_project_files()

    # Final summary
    print("\n" + "="*70)
    print("📋 INSTALLATION SUMMARY")
    print("="*70)

    if python_ok:
        print("✅ Python version: OK")
    else:
        print("⚠️  Python version: Needs update")

    if packages_ok == len(packages):
        print("✅ Required packages: All installed")
    else:
        print(f"⚠️  Required packages: {len(packages) - packages_ok} missing")
        print("   💡 Run: pip install -r requirements.txt")

    if camera_ok:
        print("✅ Camera access: Working")
    else:
        print("⚠️  Camera access: Issues detected")
        print("   💡 Check camera connection and permissions")

    if files_ok:
        print("✅ Project files: All present")
    else:
        print("⚠️  Project files: Some missing")
        print("   💡 Ensure all project files are downloaded")

    # Overall status
    overall_ok = python_ok and (packages_ok == len(packages)) and camera_ok and files_ok

    print("\n" + "="*70)
    if overall_ok:
        print("🎉 READY TO GO!")
        print("✨ Your Dr. Strange Shield System is ready to use!")
        print("🚀 Run 'python launcher.py' to start")
    else:
        print("⚠️  SETUP INCOMPLETE")
        print("🔧 Please address the issues above before running the system")
    print("="*70)

    return overall_ok

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Verification interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        print("💡 This might indicate missing dependencies")
