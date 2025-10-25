# Enhanced Dr. Strange Shield System with Improved UI
import cv2
import time
import mediapipe as mp
import numpy as np
import os
from utils import mediapipe_detection, get_center_lh,get_center_rh, points_detection, points_detection_hands
from argparse import ArgumentParser
import pickle
from datetime import datetime, timedelta
import time
import pyvirtualcam
from pyvirtualcam import PixelFormat
import signal
import sys

class DrStrangeShieldUI:
    def __init__(self):
        self.current_directory = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
        self.instruction_images = {}
        self.load_instruction_images()
        self.show_instructions = True
        self.instruction_start_time = None
        self.current_instruction = 0

    def load_instruction_images(self):
        """Load instruction images for gestures"""
        image_files = {
            'welcome': 'images/example.png',
            'key_1': 'images/position_1.png',
            'key_2': 'images/position_2.png',
            'key_3': 'images/position_3.png',
            'key_4': 'images/position_4.png'
        }

        for key, path in image_files.items():
            full_path = os.path.join(self.current_directory, path)
            if os.path.exists(full_path):
                img = cv2.imread(full_path)
                if img is not None:
                    # Resize instruction images to a standard size
                    self.instruction_images[key] = cv2.resize(img, (300, 200))
                else:
                    print(f"⚠️  Warning: Could not load {path}")
            else:
                print(f"⚠️  Warning: Image not found: {path}")

    def create_welcome_screen(self, width, height):
        """Create an attractive welcome screen"""
        screen = np.zeros((height, width, 3), dtype=np.uint8)

        # Background gradient
        for i in range(height):
            intensity = int(30 + (i / height) * 20)
            screen[i, :] = [intensity, intensity//2, intensity//3]

        # Title
        title = "DR. STRANGE SHIELD SYSTEM"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_COMPLEX, 1.2, 3)[0]
        title_x = (width - title_size[0]) // 2
        cv2.putText(screen, title, (title_x, 80), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 255), 3)

        # Subtitle
        subtitle = "Gesture-Controlled Magic Shields"
        sub_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        sub_x = (width - sub_size[0]) // 2
        cv2.putText(screen, subtitle, (sub_x, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)

        # Instructions
        instructions = [
            "INSTRUCTIONS:",
            "",
            "1. Perform KEY_1 gesture (see image)",
            "2. Quickly perform KEY_2 gesture",
            "3. Quickly perform KEY_3 gesture",
            "4. Shields will activate!",
            "5. Use KEY_4 gesture to deactivate",
            "",
            "Press SPACE to start or 'q' to quit",
            "Press 'h' anytime for help"
        ]

        y_start = 180
        for i, instruction in enumerate(instructions):
            color = (255, 255, 255) if instruction else (100, 100, 100)
            if instruction.startswith("INSTRUCTIONS:"):
                color = (0, 255, 0)
                font = cv2.FONT_HERSHEY_COMPLEX
                thickness = 2
            elif instruction.startswith(("1.", "2.", "3.", "4.", "5.")):
                color = (255, 200, 100)
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 1
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 1

            cv2.putText(screen, instruction, (50, y_start + i * 30), font, 0.6, color, thickness)

        # Add welcome image if available
        if 'welcome' in self.instruction_images:
            img = self.instruction_images['welcome']
            img_h, img_w = img.shape[:2]
            x_pos = width - img_w - 50
            y_pos = 150
            if x_pos > 0 and y_pos + img_h < height:
                screen[y_pos:y_pos+img_h, x_pos:x_pos+img_w] = img

        return screen

    def create_status_overlay(self, frame, KEY_1, KEY_2, KEY_3, SHIELDS, current_gesture=None, confidence=0.0):
        """Create an attractive status overlay"""
        height, width = frame.shape[:2]

        if SHIELDS:
            # Minimal overlay when shields are active - just show shield status
            overlay = frame.copy()
            panel_height = 60
            cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Shield status with magical effect
            shield_text = "🛡️ SHIELDS ACTIVE 🛡️"
            cv2.putText(frame, shield_text, (20, 35), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 100), 2)

            # Add magical glow effect
            cv2.putText(frame, shield_text, (19, 34), cv2.FONT_HERSHEY_COMPLEX, 0.9, (100, 255, 200), 1)

            # Deactivation hint (subtle)
            cv2.putText(frame, "Use KEY_4 gesture to deactivate", (width-280, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        else:
            # Full overlay when shields are inactive - show progress
            overlay = frame.copy()
            panel_height = 120
            cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Shield status
            shield_text = "SHIELDS: INACTIVE"
            cv2.putText(frame, shield_text, (20, 35), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 100, 255), 2)

            # Key progress with better visual feedback
            key_status = [
                ("KEY 1", KEY_1, 20, 65),
                ("KEY 2", KEY_2, 150, 65),
                ("KEY 3", KEY_3, 280, 65)
            ]

            for key_name, status, x, y in key_status:
                if status:
                    color = (0, 255, 0)
                    symbol = "✓"
                    # Add glow effect for completed keys
                    cv2.putText(frame, f"{symbol} {key_name}", (x-1, y-1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
                else:
                    color = (100, 100, 100)
                    symbol = "○"
                cv2.putText(frame, f"{symbol} {key_name}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Current gesture detection
            if current_gesture and confidence > 0.5:
                gesture_text = f"Detected: {current_gesture.upper()} ({confidence:.1%})"
                cv2.putText(frame, gesture_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Help text (always visible but subtle)
        cv2.putText(frame, "Press 'h' for help, 'q' to quit", (width-250, height-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def show_gesture_instruction(self, frame, gesture_key):
        """Show instruction for specific gesture"""
        if gesture_key in self.instruction_images:
            height, width = frame.shape[:2]
            img = self.instruction_images[gesture_key]
            img_h, img_w = img.shape[:2]

            # Position instruction image
            x_pos = width - img_w - 20
            y_pos = 140

            if x_pos > 0 and y_pos + img_h < height:
                # Add semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(overlay, (x_pos-10, y_pos-10), (x_pos+img_w+10, y_pos+img_h+10), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                # Add instruction image
                frame[y_pos:y_pos+img_h, x_pos:x_pos+img_w] = img

                # Add instruction text
                instruction_text = f"Perform {gesture_key.upper()} gesture"
                cv2.putText(frame, instruction_text, (x_pos, y_pos-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def create_help_screen(self, width, height):
        """Create help screen with all gestures"""
        screen = np.zeros((height, width, 3), dtype=np.uint8)

        # Background
        for i in range(height):
            intensity = int(20 + (i / height) * 15)
            screen[i, :] = [intensity//3, intensity//2, intensity]

        # Title
        cv2.putText(screen, "GESTURE GUIDE", (width//2-120, 50),
                   cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 255), 2)

        # Show all gesture images in a grid
        gestures = ['key_1', 'key_2', 'key_3', 'key_4']
        descriptions = [
            "KEY 1: First gesture in sequence",
            "KEY 2: Second gesture (after KEY 1)",
            "KEY 3: Final activation gesture",
            "KEY 4: Deactivation gesture"
        ]

        y_start = 100
        for i, (gesture, desc) in enumerate(zip(gestures, descriptions)):
            y_pos = y_start + i * 120

            # Add description
            cv2.putText(screen, desc, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add gesture image if available
            if gesture in self.instruction_images:
                img = cv2.resize(self.instruction_images[gesture], (200, 100))
                img_h, img_w = img.shape[:2]
                x_pos = width - img_w - 50
                screen[y_pos+10:y_pos+10+img_h, x_pos:x_pos+img_w] = img

        # Instructions
        cv2.putText(screen, "Press any key to return", (width//2-100, height-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return screen

# Enhanced main function with UI integration
def main():
    # Initialize UI
    ui = DrStrangeShieldUI()

    # Parse arguments (keeping original functionality)
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", dest="ML_model", default='models/model_svm.sav',
                        help="PATH of model FILE.", metavar="FILE")
    parser.add_argument("-t", "--threshold", dest="threshold_prediction", default=0.9, type=float,
                        help="Threshold for prediction. A number between 0 and 1. default is 0.5")
    parser.add_argument("-dc", "--det_conf", dest="min_detection_confidence", default=0.5, type=float,
                        help="Threshold for prediction. A number between 0 and 1. default is 0.5")
    parser.add_argument("-tc", "--trk_conf", dest="min_tracking_confidence", default=0.5, type=float,
                        help="Threshold for prediction. A number between 0 and 1. default is 0.5")
    parser.add_argument("-c", "--camera_id", dest="camera", default=0, type=int,
                        help="ID of the camera. An integer between 0 and N. Default is 1")
    parser.add_argument("-s", "--shield", dest="shield_video", default='effects/shield.mp4',
                        help="PATH of the video FILE.", metavar="FILE")
    parser.add_argument("-o", "--output", dest="output_mode", default='window',
                        choices=['window', 'virtual', 'both'],
                        help="Output mode: 'window' for OpenCV window only, 'virtual' for virtual camera only, 'both' for both outputs. Default is 'window'")

    args = parser.parse_args()

    # Global variables
    cap = None
    cam = None
    show_window = False

    def signal_handler(sig, frame):
        """Handle Ctrl+C interruption"""
        print("\n\n" + "="*60)
        print("\n🛑 Shutting down Dr. Strange Shield System...")
        print("🧹 Cleaning up resources...")

        if cap:
            cap.release()
            print("  ✅ Camera released")

        if show_window:
            cv2.destroyAllWindows()
            print("  ✅ OpenCV windows closed")

        if cam:
            cam.close()
            print("  ✅ Virtual camera closed")

        print("\n🏁 Application terminated successfully\n")
        print("="*60)
        sys.exit(0)

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    current_directory = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")

    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    time.sleep(2)

    # Load ML model
    try:
        model = pickle.load(open(current_directory + '/' + args.ML_model, 'rb'))
        labels = np.array(model.classes_)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Initialize variables
    KEY_1 = False
    KEY_2 = False
    KEY_3 = False
    SHIELDS = False
    scale = 1.5

    # Get camera dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load shield video
    shield = cv2.VideoCapture(current_directory + '/' + args.shield_video)
    black_screen = np.array([0,0,0])

    # Output mode settings
    show_window = args.output_mode in ['window', 'both']
    use_virtual_cam = args.output_mode in ['virtual', 'both']

    # Show welcome screen
    if show_window:
        welcome_screen = ui.create_welcome_screen(int(width * 1.5), int(height * 1.5))
        cv2.imshow('Dr. Strange Shield System', welcome_screen)

        # Wait for user to start
        print("\n" + "="*60)
        print("🛡️  DR. STRANGE SHIELD SYSTEM - ENHANCED UI")
        print("="*60)
        print("📺 Welcome screen displayed. Press SPACE to start or 'q' to quit")
        print("="*60)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to start
                break
            elif key == ord('q'):  # Quit
                cap.release()
                cv2.destroyAllWindows()
                return

    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic

    # Initialize virtual camera if needed
    if use_virtual_cam:
        try:
            cam = pyvirtualcam.Camera(width, height, 30, fmt=PixelFormat.BGR)
            print(f"🎥 Virtual Camera Device: {cam.device}")
        except Exception as e:
            print(f"⚠️  Virtual camera initialization failed: {e}")
            use_virtual_cam = False
            cam = None

    print("🚀 System Ready! Starting gesture detection...")
    print("📋 Gesture Sequence: KEY_1 → KEY_2 → KEY_3 (activate shields)")
    print("📋 Shield Deactivation: KEY_4")
    print("⌨️  Press 'h' for help, 'q' to quit")
    print("="*60 + "\n")

    # Main detection loop
    with mp_holistic.Holistic(min_detection_confidence=args.min_detection_confidence,
                              min_tracking_confidence=args.min_tracking_confidence,
                              model_complexity=0) as holistic:

        show_help = False
        current_gesture = None
        gesture_confidence = 0.0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Handle help screen
                if show_help:
                    help_screen = ui.create_help_screen(int(width * 1.5), int(height * 1.5))
                    if show_window:
                        cv2.imshow('Dr. Strange Shield System', help_screen)

                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:  # Any key pressed
                        show_help = False
                    continue

                # Load shield frame
                ret_shield, frame_shield = shield.read()
                if not ret_shield:
                    shield = cv2.VideoCapture(current_directory + '/' + args.shield_video)
                    ret_shield, frame_shield = shield.read()

                # Make detection
                frame, results = mediapipe_detection(frame, holistic)
                xMinL, xMaxL, yMinL, yMaxL = get_center_lh(frame, results)
                xMinR, xMaxR, yMinR, yMaxR = get_center_rh(frame, results)

                # Shield effect processing (keeping original logic)
                mask = cv2.inRange(frame_shield, black_screen, black_screen)
                res = cv2.bitwise_and(frame_shield, frame_shield, mask=mask)
                res = frame_shield - res
                alpha = 1

                # Apply shield effects for left hand
                if SHIELDS and xMinL:
                    xc_lh = (xMaxL+xMinL)/2
                    yc_lh = (yMaxL+yMinL)/2
                    xc_lh = int(width*xc_lh)
                    yc_lh = int(height*yc_lh)

                    l_width_shield = int(width*(xMaxL-xMinL)/2*3.5*scale)
                    l_height_shield = int(height*(yMaxL-yMinL)/2*2*scale)

                    res2 = cv2.resize(res, (l_width_shield*2, l_height_shield*2))

                    start_h = 0
                    start_w = 0
                    stop_h = l_height_shield*2
                    stop_w = l_width_shield*2

                    f_start_h = yc_lh-l_height_shield
                    f_stop_h = yc_lh+l_height_shield
                    f_start_w = xc_lh-l_width_shield
                    f_stop_w = xc_lh+l_width_shield

                    if yc_lh-l_height_shield < 0:
                        start_h = -yc_lh+l_height_shield
                        f_start_h = 0
                    if yc_lh+l_height_shield > height:
                        stop_h = l_height_shield + height - yc_lh
                        f_stop_h = height
                    if xc_lh-l_width_shield < 0:
                        start_w = -xc_lh+l_width_shield
                        f_start_w = 0
                    if xc_lh+l_width_shield > width:
                        stop_w = l_width_shield + width - xc_lh
                        f_stop_w = width

                    res2 = res2[start_h:stop_h, start_w:stop_w,:]
                    frame_shield = cv2.addWeighted(frame[f_start_h:f_stop_h,f_start_w:f_stop_w], alpha, res2, 1, 1, frame)
                    frame[f_start_h:f_stop_h,f_start_w:f_stop_w] = frame_shield

                # Apply shield effects for right hand
                if SHIELDS and xMinR:
                    xc_rh = (xMaxR+xMinR)/2
                    yc_rh = (yMaxR+yMinR)/2
                    xc_rh = int(width*xc_rh)
                    yc_rh = int(height*yc_rh)

                    r_width_shield = int(width*(xMaxR-xMinR)/2*3.5*scale)
                    r_height_shield = int(height*(yMaxR-yMinR)/2*2*scale)

                    res3 = cv2.resize(res, (r_width_shield*2, r_height_shield*2))

                    start_h = 0
                    start_w = 0
                    stop_h = r_height_shield*2
                    stop_w = r_width_shield*2

                    f_start_h = yc_rh-r_height_shield
                    f_stop_h = yc_rh+r_height_shield
                    f_start_w = xc_rh-r_width_shield
                    f_stop_w = xc_rh+r_width_shield

                    if yc_rh-r_height_shield < 0:
                        start_h = -yc_rh+r_height_shield
                        f_start_h = 0
                    if yc_rh+r_height_shield > height:
                        stop_h = r_height_shield + height - yc_rh
                        f_stop_h = height
                    if xc_rh-r_width_shield < 0:
                        start_w = -xc_rh+r_width_shield
                        f_start_w = 0
                    if xc_rh+r_width_shield > width:
                        stop_w = r_width_shield + width - xc_rh
                        f_stop_w = width

                    res3 = res3[start_h:stop_h, start_w:stop_w,:]
                    frame_shield = cv2.addWeighted(frame[f_start_h:f_stop_h,f_start_w:f_stop_w], alpha, res3, 1, 1, frame)
                    frame[f_start_h:f_stop_h,f_start_w:f_stop_w] = frame_shield

                # Gesture recognition logic (keeping original)
                current_gesture = None
                gesture_confidence = 0.0

                if xMinL and xMinR and SHIELDS:
                    prediction = model.predict(np.array([points_detection_hands(results)]))[0]
                    pred_prob = np.max(model.predict_proba(np.array([points_detection_hands(results)])))
                    current_gesture = prediction
                    gesture_confidence = pred_prob

                    if (prediction == 'key_4') and (pred_prob > 0.85):
                        KEY_1 = False
                        KEY_2 = False
                        KEY_3 = False
                        SHIELDS = False

                elif xMinL and xMinR and (not SHIELDS):
                    prediction = model.predict(np.array([points_detection_hands(results)]))[0]
                    pred_prob = np.max(model.predict_proba(np.array([points_detection_hands(results)])))
                    current_gesture = prediction
                    gesture_confidence = pred_prob

                    if (prediction == 'key_1') and (pred_prob > 0.85):
                        t1 = datetime.now()
                        KEY_1 = True
                    elif (prediction == 'key_2') and (pred_prob > 0.85) and KEY_1:
                        t2 = datetime.now()
                        if t1 + timedelta(seconds=2) > t2:
                            KEY_2 = True
                        else:
                            KEY_1 = False
                            KEY_2 = False
                    elif (prediction == 'key_3') and (pred_prob > 0.85) and KEY_1 and KEY_2:
                        t3 = datetime.now()
                        if t2 + timedelta(seconds=2) > t3:
                            KEY_3 = True
                            SHIELDS = True
                        else:
                            KEY_1 = False
                            KEY_2 = False

                # Add UI overlays
                frame = ui.create_status_overlay(frame, KEY_1, KEY_2, KEY_3, SHIELDS, current_gesture, gesture_confidence)

                # Show gesture instructions
                if not SHIELDS:
                    if not KEY_1:
                        frame = ui.show_gesture_instruction(frame, 'key_1')
                    elif not KEY_2:
                        frame = ui.show_gesture_instruction(frame, 'key_2')
                    elif not KEY_3:
                        frame = ui.show_gesture_instruction(frame, 'key_3')
                else:
                    frame = ui.show_gesture_instruction(frame, 'key_4')

                # Display frame
                if show_window:
                    # Resize frame for bigger display
                    display_frame = cv2.resize(frame, (int(width * 1.5), int(height * 1.5)))
                    cv2.imshow('Dr. Strange Shield System', display_frame)

                # Send to virtual camera
                if use_virtual_cam and cam:
                    cam.send(frame)
                    cam.sleep_until_next_frame()

                # Handle keyboard input
                if show_window:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('h'):
                        show_help = True
                else:
                    time.sleep(0.033)  # ~30 FPS

        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted - Shutting down...")
        except Exception as e:
            print(f"\n❌ Error during execution: {e}")
        finally:
            # Final cleanup
            print("\n\n" + "="*60)
            print("\n🧹 Final cleanup...\n")
            if cap:
                cap.release()
                print("  ✅ Camera released")
            if show_window:
                cv2.destroyAllWindows()
                print("  ✅ OpenCV windows closed")
            if cam:
                cam.close()
                print("  ✅ Virtual camera closed")
            print("\n🏁 Application terminated\n")

if __name__ == "__main__":
    main()
