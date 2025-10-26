# 🛡️ Dr. Strange Shields - Gesture Control System

A real-time hand gesture recognition system that creates magical shield effects inspired by Doctor Strange, with **background blur** like Zoom/Teams!

<br>
<p align="center">
  <img width="640"  src="./images/example.png">
</p>
<br>

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🚀 Quick Start (3 Steps!)

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the System

```bash
# Easy launcher with menu (Recommended)
python launcher.py

# OR directly run with background blur
python shield_with_blur.py
```

### 3️⃣ Perform Gestures

1. **KEY_1** → **KEY_2** → **KEY_3** = Activate shields! 🛡️
2. **KEY_4** = Deactivate shields

**That's it!** ✨

---

## 🎯 Available Versions

| Version                | Command                      | Features                        |
| ---------------------- | ---------------------------- | ------------------------------- |
| 🎭 **Background Blur** | `python shield_with_blur.py` | Zoom-like blur + shields        |
| ✨ **Enhanced UI**     | `python shield_enhanced.py`  | Better interface + instructions |
| 🔧 **Original**        | `python shield.py -o window` | Basic version                   |
| 📋 **Launcher**        | `python launcher.py`         | Interactive menu                |

## 🎮 Controls

- **SPACE**: Start system from welcome screen
- **H**: Show help with gesture images
- **Q**: Quit application
- **Ctrl+C**: Emergency exit

## 🔑 Gesture Sequence

<br>
<p align="center">
  <img width="320"  src="./images/position_1.png">
  <img width="320"  src="./images/position_2.png">
  <img width="320"  src="./images/position_3.png">
</p>
<br>

**Activation:** KEY_1 → KEY_2 → KEY_3 (within 3 seconds each)

<br>
<p align="center">
  <img width="360"  src="./images/position_4.png">
</p>
<br>

**Deactivation:** KEY_4

## ✨ Features

### 🎭 Background Blur Version (NEW!)

- **Zoom-like Background Blur**: Professional video call appearance
- **Person Detection**: You stay sharp, background blurs
- **Adjustable Blur**: Control blur intensity
- **Real-time Processing**: Smooth 30 FPS performance

### 🛡️ Shield Effects

- **Magical Shields**: Appear on your hands when activated
- **Real-time Tracking**: Shields follow hand movements
- **Visual Effects**: Glowing borders and magical animations
- **Gesture Recognition**: ML-powered hand gesture detection

### 🎨 Enhanced UI

- **Welcome Screen**: Beautiful startup interface
- **Visual Instructions**: Shows gesture images in real-time
- **Progress Tracking**: See which keys you've activated
- **Help System**: Press 'h' for gesture guide

## 🔧 Advanced Usage

### Custom Blur Intensity

```bash
python shield_with_blur.py -b 75  # Higher = more blur
```

### Different Camera

```bash
python shield_with_blur.py -c 1  # Use camera ID 1
```

### Virtual Camera Output

```bash
python shield_enhanced.py -o virtual  # Requires OBS
```

## 📦 Installation Details

### Requirements

- Python 3.9+
- Webcam/Camera
- Windows/Mac/Linux

### Dependencies

The system uses these main packages:

- **opencv-python**: Camera and image processing
- **mediapipe**: Hand tracking and person segmentation
- **scikit-learn**: Gesture classification
- **numpy**: Numerical operations
- **pyvirtualcam**: Virtual camera support (optional)

### Verify Installation

```bash
python verify_installation.py
```

## 🐛 Troubleshooting

### Camera Issues

```bash
# Try different camera ID
python shield_with_blur.py -c 1

# Check available cameras
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

### Performance Issues

```bash
# Lower blur intensity for better performance
python shield_with_blur.py -b 25

# Use original version (no blur)
python shield.py -o window
```

### Gesture Not Detected

- Ensure both hands are visible
- Good lighting conditions
- Clear background
- Perform gestures slowly and clearly

### Virtual Camera Not Working

- Install OBS Studio with Virtual Camera
- Use window mode instead: `-o window`

## 🎬 Demo

The system works in real-time:

1. **Background Blur**: Your background automatically blurs (like Zoom)
2. **Gesture Detection**: Perform the 4 key gestures
3. **Shield Activation**: Magical shields appear on your hands
4. **Clean View**: When shields are active, UI disappears for immersive experience

## 📁 Project Structure

```
dr-strange-shields/
├── shield_with_blur.py      # Main app with background blur
├── shield_enhanced.py       # Enhanced UI version
├── shield.py               # Original version
├── launcher.py             # Interactive launcher
├── utils.py                # Utility functions
├── requirements.txt        # Dependencies
├── models/
│   └── model_svm.sav      # Trained gesture model
├── effects/
│   └── shield.mp4         # Shield video effect
├── images/
│   ├── position_1.png     # Gesture instruction images
│   ├── position_2.png
│   ├── position_3.png
│   └── position_4.png
└── README.md
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] More gesture types
- [ ] Custom shield effects
- [ ] Sound effects
- [ ] Mobile app version
- [ ] Web version

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe**: Google's ML framework
- **OpenCV**: Computer vision library
- **Marvel Studios**: Inspiration from Doctor Strange

---

**Made with ✨ magic and 🐍 Python**

_Transform your video calls with magical shields and professional background blur!_
