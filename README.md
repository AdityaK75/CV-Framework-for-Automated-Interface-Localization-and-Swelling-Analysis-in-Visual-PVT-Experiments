# PVT Swelling Analyzer

**Designed & Developed by [Aditya Kanagalekar](https://github.com/AdityaK75)**

Welcome to the PVT Swelling Analyzer! This is a web-based software tool designed to automatically measure fluid volumes and swelling factors in high-pressure visual PVT (Pressure-Volume-Temperature) experiments. 

Even if the images of your fluid have heavy glare, reflections, or are hard to see, this software uses advanced image processing to perfectly find the edge of the fluid and calculate how much it has expanded.

---

## 🌟 Key Features

* **Supports Multiple Reactor Types:** Works with both standard Tubular (sight-glass) reactors and Spherical reactors.
* **Automatic Detection:** Automatically finds the exact fluid line (meniscus) without human guessing.
* **Manual Override:** If the glare is absolutely too extreme, you can manually click the fluid line to guide the software.
* **Report Generation:** Automatically generates and saves professional, watermarked measurement reports with all your experimental data (Temperature, Pressure, Solvent used, Time elapsed, etc.).

---

## 🧩 The Main Components

The software is built to be simple to use, but under the hood, it is divided into three main parts:

1. **The Brain (The Engine):** (`src/pvt_analyzer.py` & `src/vision_engine.py`)
   This is the mathematical core of the program. It takes your raw images, applies specialized filters to strip away glare, and traces the exact physical shape of the fluid. It then calculates the precise height and volume of the liquid based on your reactor's physical dimensions.

2. **The Server (The Bridge):** (`src/web_gui.py`)
   This runs invisibly in the background on your computer. It listens to what you click on the website, sends your images to the "Brain" for analysis, and then perfectly packages the results to send back to your screen.

3. **The User Dashboard (The Frontend):** (`templates/auto_swelling.html`)
   This is the sleek, modern website you interact with in your browser. It lets you easily upload images, draw boundaries, enter your experimental data, and view the generated visual reports.

---

## 🚀 How to Set Up (For Beginners)

You don't need to be a programmer to run this software! Just follow these easy steps:

### Prerequisites
You need to have **Python** installed on your computer. If you don't have it, download it for free from [python.org](https://www.python.org/downloads/) and run the installer.

### Step 1: Open your Terminal / Command Prompt
Open your computer's Terminal (if you are on a Mac) or Command Prompt (if you are on Windows). Navigate to the folder where you downloaded this project.

### Step 2: Install the Required Tools
We need to download the background tools the software relies on (like image processing libraries). Run this command and wait for it to finish:
```bash
pip install -r requirements.txt
```

### Step 3: Start the Application
Once the installation is done, start the software by running:
```bash
python src/web_gui.py
```

### Step 4: Open in Your Browser
The terminal will display a message saying the server is running. Open your favorite web browser (like Chrome, Safari, or Edge) and go to this web address:
**http://localhost:5006**

---

## 📖 Quick Start Guide

Using the application requires just a few simple steps to guarantee scientific accuracy:

1. **Upload Images:** Click the "Add" button to load your experiment images. They will appear in the gallery at the bottom.
2. **Select Geometry:** At the top left, choose whether your experiment uses a **Tubular** or **Spherical** reactor.
3. **Calibrate the Scale:** 
   * *For Tubular Reactors:* Click the top and bottom physical edges of the sight-glass window in the image, and tell the software the actual physical distance between them in millimeters.
   * *For Spherical Reactors:* Simply type in the physical diameter of the sphere.
4. **Enter Data:** Type in your sample name, solvent, pressure, and temperature so they get recorded on your report.
5. **Auto-Detect:** Click the big purple "AUTO-DETECT & COMPUTE" button. The software will process the image and show you the exact volume!
6. **View Reports:** All processed images and measurements are automatically saved as professional, watermarked reports in the `results/` folder on your computer.
