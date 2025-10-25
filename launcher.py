#!/usr/bin/env python3
"""
Dr. Strange Shield System Launcher
Enhanced version with improved UI and instructions
"""

import os
import sys
import subprocess

def print_banner():
    print("\n" + "="*70)
    print("🛡️  DR. STRANGE SHIELD SYSTEM - LAUNCHER")
    print("="*70)
    print("✨ Enhanced version with improved UI and gesture instructions")
    print("📸 Uses images from the 'images' folder for visual guidance")
    print("🎯 Real-time gesture recognition with visual feedback")
    print("="*70)

def print_menu():
    print("\n📋 AVAILABLE OPTIONS:")
    print("="*50)
    print("1. 🖥️  Launch with OpenCV Window (Recommended)")
    print("2. 📹 Launch with Virtual Camera")
    print("3. 🔄 Launch with Both Window & Virtual Camera")
    print("4. ⚙️  Launch Original Version")
    print("5. 📖 View Help & Instructions")
    print("6. ❌ Exit")
    print("="*50)

def show_help():
    print("\n" + "="*70)
    print("📖 DR. STRANGE SHIELD SYSTEM - HELP")
    print("="*70)
    print("\n🎯 HOW IT WORKS:")
    print("• The system uses your camera to detect hand gestures")
    print("• Perform a sequence of gestures to activate magical shields")
    print("• Visual instructions are shown on screen with example images")
    print("\n🔑 GESTURE SEQUENCE:")
    print("1. KEY_1: First gesture (see position_1.png)")
    print("2. KEY_2: Second gesture (see position_2.png)")
    print("3. KEY_3: Third gesture (see position_3.png)")
    print("4. 🛡️  SHIELDS ACTIVATE!")
    print("5. KEY_4: Deactivation gesture (see position_4.png)")
    print("\n⌨️  CONTROLS:")
    print("• SPACE: Start the system from welcome screen")
    print("• H: Show help screen during operation")
    print("• Q: Quit the application")
    print("• Ctrl+C: Emergency exit")
    print("\n📁 REQUIREMENTS:")
    print("• Camera connected and working")
    print("• All dependencies installed (run: pip install -r requirements.txt)")
    print("• Images folder with instruction images")
    print("• Models folder with trained ML model")
    print("• Effects folder with shield video")
    print("\n🔧 TROUBLESHOOTING:")
    print("• If camera doesn't work, try changing camera ID with -c parameter")
    print("• If virtual camera fails, use window-only mode")
    print("• Make sure all image files exist in the images folder")
    print("="*70)

def run_command(cmd):
    """Run a command and handle errors"""
    try:
        print(f"\n🚀 Launching: {' '.join(cmd)}")
        print("="*50)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running command: {e}")
        print("💡 Try installing dependencies: pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

def main():
    print_banner()

    while True:
        print_menu()

        try:
            choice = input("\n🎯 Enter your choice (1-6): ").strip()

            if choice == '1':
                print("\n✅ Launching Enhanced Version with OpenCV Window...")
                run_command([sys.executable, "shield_enhanced.py", "-o", "window"])

            elif choice == '2':
                print("\n✅ Launching Enhanced Version with Virtual Camera...")
                print("⚠️  Note: Requires OBS or compatible virtual camera software")
                run_command([sys.executable, "shield_enhanced.py", "-o", "virtual"])

            elif choice == '3':
                print("\n✅ Launching Enhanced Version with Both Outputs...")
                print("⚠️  Note: Requires OBS or compatible virtual camera software")
                run_command([sys.executable, "shield_enhanced.py", "-o", "both"])

            elif choice == '4':
                print("\n✅ Launching Original Version...")
                run_command([sys.executable, "shield.py", "-o", "window"])

            elif choice == '5':
                show_help()
                input("\n📖 Press Enter to return to menu...")

            elif choice == '6':
                print("\n👋 Goodbye! May the mystic arts be with you!")
                break

            else:
                print("\n❌ Invalid choice. Please enter a number between 1-6.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! May the mystic arts be with you!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
