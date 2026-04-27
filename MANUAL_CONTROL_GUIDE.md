# PVT Manual Control - Complete Guide

## 📋 Overview

You now have **TWO manual control interfaces** for meniscus detection:

| Interface | Type | Launch | Access |
|-----------|------|--------|--------|
| **Desktop GUI** | Tkinter (native macOS) | `python3 src/main_gui_manual.py` | Direct window |
| **Web GUI** | Flask + HTML5 Canvas | `python3 src/web_gui_manual.py` | `http://localhost:5003` |

Both GUIs provide the same workflow: manual ROI + calibration, then you set a meniscus middle point (meniscus line is forced through it) and the GUI computes water-column height.

---

## 🚀 Quick Start

### Desktop GUI (Tkinter)
```bash
cd /Users/adityakanagalekar/BTP
source .venv/bin/activate
python3 src/main_gui_manual.py
```
**Result**: Native macOS window opens on your desktop

### Web GUI (Flask)
```bash
cd /Users/adityakanagalekar/BTP
source .venv/bin/activate
python3 src/web_gui_manual.py
```
**Then**: Open browser to `http://localhost:5003`

---

## 📝 Manual Setup + Auto Detection (Both GUIs)

### Step 1: Load Image
- **Desktop**: Click "📁 Select Image" or "📂 Select Folder"
- **Web**: Click file input, select image, click "📁 Load Image"
- Status shows: ✅ Loaded (width×height)px

### Step 2: Select Rectangle Area
- Click **"🔲 Draw Rectangle"**
- **On canvas**: Click and drag to draw rectangle around measurement area
- Rectangle appears in **lime green**
- Status shows: ✅ Rectangle (x1,y1)→(x2,y2), Size: W×H

### Step 3: Calibration (92 mm)
1. **Distance field**: Pre-filled with 92.0 mm (adjust if needed)
2. Click **"🔺 Click TOP Point"**
   - Click on TOP of your reference object (appears as **yellow circle**)
3. Click **"🔽 Click BOTTOM Point"**
   - Click on BOTTOM of reference object (appears as **cyan circle**)
   - These points span 92 mm (or your custom distance)
4. Status shows: ✅ Calibrated with TOP/BOTTOM coordinates and distance

### Step 4: Set Meniscus Middle Point
1. Click **"🎯 Click Meniscus Middle"** on the liquid surface center (this is where the meniscus line will be)
2. Click **"▶ COMPUTE HEIGHT"**
3. **Result**:
   - Image redraws with the meniscus line through your selected middle point
   - Water height calculated: `(bottom_Y - middle_Y) × mm_per_pixel`
   - Output saved to `results/` folder
   - Status shows: ✅ Meniscus Y and Water Height in `mm`

---

## 🎨 Visual Indicators on Canvas

### Desktop GUI (Tkinter Canvas)
```
Lime rectangle     = Your ROI area
Yellow circle      = TOP calibration point
Cyan circle        = BOTTOM calibration point
Orange dashed line = Calibration distance
Orange horizontal  = Final detected meniscus line
```

### Web GUI (HTML5 Canvas)
Identical visual indicators + resizable canvas that scales with window

---

## 💾 Save & Load Setup

Both GUIs support saving and loading complete setups:

### Save Setup
- Click **"💾 Save"**
- All selections saved to JSON: `results/setup_N.json`
- Contains: ROI, calibration points, detected meniscus, mm distance

### Load Setup
- Click **"📂 Load"** (Web: enter file path)
- Reload previous configuration instantly
- All points redraw on canvas

### Reset
- Click **"🗑️ Reset"**
- Clear all selections
- Start fresh with same image

---

## 📊 Example Workflow

### Desktop
```
1. Launch: python3 src/main_gui_manual.py
2. Window appears → Click "📁 Select Image"
3. Choose Img1.png from imgdata/water/
4. Image displays in canvas
5. Click "🔲 Draw Rectangle" → Draw on image
6. Click "🔺 Click TOP Point" → Click top reference
7. Click "🔽 Click BOTTOM Point" → Click bottom reference
8. Calibration shows: ✅ 92.0mm = 1600px
9. Click "🎯 Click Meniscus Middle"
10. Click "▶ COMPUTE HEIGHT"
11. Result: Water Height: 57.33 mm (saved to results/Img1_manual_middle.png)
```

### Web
```
1. Launch: python3 src/web_gui_manual.py
2. Browser: Navigate to http://localhost:5003
3. Upload image using file input
4. Canvas shows image
5. Follow the manual ROI + calibration steps, click "Meniscus Middle", then click "COMPUTE HEIGHT"
6. Results display in status box + image updated in canvas
```

---

## 🔧 Customization

### Change Distance Reference
1. Edit **Distance (mm)** field in Step 3
2. Default: 92.0 mm
3. Your calibration points define the height, you define the mm value

### Detection Method
There is no detection method choice in this version: the meniscus is forced through the clicked middle point and the water height is computed directly.

---

## 📂 File Locations

```
/Users/adityakanagalekar/BTP/
├── src/
│   ├── main_gui_manual.py      ← Desktop GUI (Tkinter)
│   ├── web_gui_manual.py       ← Web GUI (Flask)
│   ├── pvt_analyzer.py         ← Core analyzer (all methods)
├── templates/
│   └── manual_control.html     ← Web UI template
├── results/                    ← Output images & CSV
├── imgdata/water/              ← Test images
└── MANUAL_GUI_GUIDE.md         ← This file
```

---

## 🐛 Troubleshooting

### Desktop GUI doesn't appear
- **macOS**: Make sure no other Tkinter window is open
- Try: Close all Python processes, then relaunch
- Check terminal for error messages

### Web GUI shows "Connection refused"
- Make sure Flask server is running: `python3 src/web_gui_manual.py`
- Check port 5003 isn't blocked
- If port is in use, modify port in `src/web_gui_manual.py`

### Image not loading
- Ensure image is PNG/JPG/BMP
- Check file permissions
- Try absolute file path instead of relative

### Drawing not working
- **Desktop**: Click button FIRST, then draw on canvas
- **Web**: Click button, mode indicator changes, then click on canvas
- Ensure canvas is visible and focused

### Water height looks wrong (e.g., 0 mm)
- Make sure you clicked BOTH calibration points (top & bottom)
- Meniscus middle point should fall BETWEEN the top and bottom calibration points
- Check mm value isn't 0

---

## 📈 Expected Output

After detection, you'll see:
```
results/
├── Img1_manual_middle.png        ← Annotated image with meniscus line
├── Img2_manual_middle.png
├── setup_0.json                    ← Saved configuration
└── ...
```

Each PNG shows:
- Original image
- Green rectangle (ROI)
- Yellow/cyan circles (calibration)
- Orange horizontal line (meniscus line)
- Water height text (e.g., "Height: 57.33 mm" depending on overlay wording)

---

## ⌨️ Keyboard Shortcuts

### Desktop GUI
- **⌘Q**: Quit application
- **Prev/Next buttons**: Navigate through folder of images

### Web GUI
- **Ctrl+R**: Refresh browser
- File upload: Standard file picker (OS-dependent)

---

## 🎯 Tips & Best Practices

1. **Always calibrate first** - Height depends on calibration scale
2. **Click accurately** - Use crosshair cursor as guide
3. **Save setup** - Reuse calibration for multiple images
4. **Use literature method** - Best accuracy (sub-pixel precision)
5. **ROI is required** - detection uses the rectangle you draw
6. **Batch processing** - For CLI batch runs, you must provide `--roi` coordinates

---

## 🔄 Batch Processing via CLI

Once you've created a setup with manual control GUI, use it for batch CLI processing:

```bash
# Process multiple images with saved setup
python3 src/pvt_analyzer.py \
  --image imgdata/water/ \
  --batch \
  --method literature \
  --roi 100 2000 1500 1700 \
  --out results \
  --verbose
```

---

## 📞 Support

For issues:
1. Check error message in terminal/console
2. Verify image file exists and is readable
3. Ensure virtualenv is activated
4. Try resetting all selections and starting fresh

---

**Version**: Manual Control v2.0 (Desktop + Web)  
**Last Updated**: March 2026  
**Status**: ✅ Production Ready
