#!/usr/bin/env python3
"""
Demo script to showcase the enhanced UI without requiring camera
"""

import cv2
import numpy as np
import os
import sys

# Add the current directory to path to import from shield_enhanced
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shield_enhanced import DrStrangeShieldUI

def demo_ui():
    print("🎬 Dr. Strange Shield System - UI Demo")
    print("="*50)
    print("This demo shows the enhanced UI screens without requiring a camera")
    print("Press any key to cycle through different screens, 'q' to quit")
    print("="*50)

    # Initialize UI
    ui = DrStrangeShieldUI()

    # Demo dimensions (simulating camera resolution)
    width, height = 640, 480
    display_width, display_height = int(width * 1.5), int(height * 1.5)

    screens = [
        ("Welcome Screen", lambda: ui.create_welcome_screen(display_width, display_height)),
        ("Help Screen", lambda: ui.create_help_screen(display_width, display_height)),
    ]

    # Create demo frames with different states
    demo_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Add some fake content to demo frame
    cv2.putText(demo_frame, "CAMERA FEED SIMULATION", (width//2-150, height//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    status_screens = [
        ("Status Overlay - Inactive", lambda: ui.create_status_overlay(
            demo_frame.copy(), False, False, False, False)),
        ("Status Overlay - KEY 1 Active", lambda: ui.create_status_overlay(
            demo_frame.copy(), True, False, False, False, "key_1", 0.92)),
        ("Status Overlay - KEY 1+2 Active", lambda: ui.create_status_overlay(
            demo_frame.copy(), True, True, False, False, "key_2", 0.88)),
        ("Status Overlay - Shields Active", lambda: ui.create_status_overlay(
            demo_frame.copy(), True, True, True, True, "key_4", 0.75)),
    ]

    # Add gesture instruction demos
    instruction_screens = []
    for gesture in ['key_1', 'key_2', 'key_3', 'key_4']:
        instruction_screens.append((
            f"Gesture Instruction - {gesture.upper()}",
            lambda g=gesture: ui.show_gesture_instruction(demo_frame.copy(), g)
        ))

    all_screens = screens + status_screens + instruction_screens
    current_screen = 0

    print(f"\n🎯 Starting demo with {len(all_screens)} screens...")
    print("Use LEFT/RIGHT arrow keys or SPACE to navigate, 'q' to quit\n")

    while True:
        screen_name, screen_func = all_screens[current_screen]
        screen = screen_func()

        # Resize for better visibility
        if screen.shape[:2] != (display_height, display_width):
            screen = cv2.resize(screen, (display_width, display_height))

        # Add screen info
        info_text = f"Screen {current_screen + 1}/{len(all_screens)}: {screen_name}"
        cv2.putText(screen, info_text, (10, display_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Dr. Strange Shield System - UI Demo', screen)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == 83 or key == ord(' '):  # Right arrow or space
            current_screen = (current_screen + 1) % len(all_screens)
        elif key == 81:  # Left arrow
            current_screen = (current_screen - 1) % len(all_screens)
        elif key == 27:  # Escape
            break

    cv2.destroyAllWindows()
    print("\n✨ Demo completed! The enhanced UI is ready to use.")
    print("🚀 Run 'python launcher.py' to start the full system")

if __name__ == "__main__":
    try:
        demo_ui()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
