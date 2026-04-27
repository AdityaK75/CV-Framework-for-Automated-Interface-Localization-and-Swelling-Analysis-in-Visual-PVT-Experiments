#!/usr/bin/env python3
"""Diagnostic script to check GUI environment and dependencies."""

import sys
import os

print("=== Python & Environment ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
print(f"DISPLAY env: {os.environ.get('DISPLAY', 'NOT SET')}")

print("\n=== Importing Core Libraries ===")
try:
    import tkinter
    print(f"✓ Tkinter imported (version: {tkinter.TkVersion})")
except ImportError as e:
    print(f"✗ Tkinter failed: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"✓ OpenCV imported (version: {cv2.__version__})")
except ImportError as e:
    print(f"✗ OpenCV failed: {e}")

try:
    from PIL import Image, ImageTk
    print("✓ PIL/Pillow imported")
except ImportError as e:
    print(f"✗ PIL/Pillow failed: {e}")

try:
    from pvt_analyzer import PVTAnalyzer
    print("✓ PVTAnalyzer imported")
except ImportError as e:
    print(f"✗ PVTAnalyzer failed: {e}")

print("\n=== GUI Creation Test ===")
try:
    # Try to create a minimal Tk window without mainloop
    root = tkinter.Tk()
    print("✓ Tk root created")
    
    root.title("Test Window")
    root.geometry("300x200")
    print("✓ Tk window configured")
    
    # Try to update without entering mainloop
    root.update_idletasks()
    print("✓ Tk window updated (should be visible)")
    
    # Clean up
    root.destroy()
    print("✓ Tk window destroyed")
    
except Exception as e:
    print(f"✗ Tk window creation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Summary ===")
print("If all checks pass but GUI window doesn't appear:")
print("- You may be on a headless or SSH session without X11 forwarding")
print("- Try running: open -a Terminal && source .venv/bin/activate && python3 src/main_gui.py")
print("- Or check System Preferences > Security & Privacy for Terminal permissions")
