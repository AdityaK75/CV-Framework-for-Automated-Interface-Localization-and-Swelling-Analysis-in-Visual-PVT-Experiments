"""
PVT Meniscus GUI - Manual Setup + Auto Detection.

Manual steps:
1. Select rectangle ROI
2. Select top and bottom calibration points
3. Run automatic meniscus detection
"""
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import json
import os
import logging
import cv2
import numpy as np
from pvt_analyzer import PVTAnalyzer


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class ManualControlGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('PVT Meniscus - Manual Control')
        self.geometry('1500x900')
        
        self.analyzer = PVTAnalyzer()
        self.image_paths = []
        self.current_index = 0
        self.current_image = None  # image currently shown on canvas
        self.original_image = None  # pristine source image for rerendering/processing
        self.display_image = None  # PIL for display
        self.tk_image = None
        
        # User-selected control points
        self.roi_rect = None  # (x1, y1, x2, y2)
        self.calib_top_point = None  # (x, y) in image coords
        self.calib_bottom_point = None  # (x, y) in image coords
        self.meniscus_middle_point = None  # (x, y) in image coords (what the meniscus lies on)
        self.detected_meniscus_y = None  # detected Y coordinate in image coords
        self.mm_distance = 92.0
        self.is_calibrated = False
        
        # Drawing state
        self._drawing_roi = False
        self._drawing_calib = False
        self._drawing_meniscus_middle = False
        self._drag_start = None
        self._rect_canvas_id = None
        
        # ============= LEFT PANEL: CONTROLS & STATUS =============
        left_panel = tk.Frame(self, width=350, bg='#f5f5f5')
        left_panel.pack(side='left', fill='both', padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        # Title
        tk.Label(left_panel, text='Manual Control Panel', 
                font=('Arial', 14, 'bold'), bg='#f5f5f5').pack(fill='x', padx=5, pady=(10, 5))
        
        # ===== STEP 1: Image =====
        self._create_step_header(left_panel, '1️⃣ LOAD IMAGE')
        tk.Button(left_panel, text='📁 Select Image', command=self.load_image,
                 bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'), width=25).pack(fill='x', padx=5, pady=2)
        tk.Button(left_panel, text='📂 Select Folder', command=self.load_folder,
                 bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'), width=25).pack(fill='x', padx=5, pady=2)
        
        frame_nav = tk.Frame(left_panel, bg='#f5f5f5')
        frame_nav.pack(fill='x', padx=5, pady=2)
        tk.Button(frame_nav, text='◀ Prev', command=self.prev_image, width=12).pack(side='left', padx=2)
        tk.Button(frame_nav, text='Next ▶', command=self.next_image, width=12).pack(side='left', padx=2)
        
        self.label_image = tk.Label(left_panel, text='No image loaded', 
                                   font=('Arial', 9), bg='#fff3cd', fg='#856404', 
                                   wraplength=300, justify='left')
        self.label_image.pack(fill='x', padx=5, pady=5)
        
        # ===== STEP 2: Rectangle ROI =====
        self._sep(left_panel)
        self._create_step_header(left_panel, '2️⃣ SELECT RECTANGLE AREA')
        tk.Button(left_panel, text='🔲 Draw Rectangle on Image', command=self.start_draw_roi,
                 bg='#2196F3', fg='white', font=('Arial', 10, 'bold'), width=25).pack(fill='x', padx=5, pady=2)
        tk.Label(left_panel, text='(Click & drag on canvas)', font=('Arial', 8), 
                bg='#f5f5f5').pack(fill='x', padx=5, pady=1)
        
        self.label_roi = tk.Label(left_panel, text='🔴 No rectangle selected', 
                                 font=('Arial', 9), bg='#fff3cd', fg='#856404', 
                                 wraplength=300, justify='left')
        self.label_roi.pack(fill='x', padx=5, pady=5)
        
        # ===== STEP 3: Calibration =====
        self._sep(left_panel)
        self._create_step_header(left_panel, '3️⃣ CALIBRATION (92 mm)')
        
        frame_mm = tk.Frame(left_panel, bg='#f5f5f5')
        frame_mm.pack(fill='x', padx=5, pady=2)
        tk.Label(frame_mm, text='Distance (mm):', font=('Arial', 9), bg='#f5f5f5').pack(side='left', padx=2)
        self.mm_var = tk.DoubleVar(value=92.0)
        tk.Entry(frame_mm, textvariable=self.mm_var, font=('Arial', 10), width=10).pack(side='left', padx=2)
        
        tk.Button(left_panel, text='🔺 Click TOP Point', command=self.start_calib_top,
                 bg='#FF9800', fg='white', font=('Arial', 10, 'bold'), width=25).pack(fill='x', padx=5, pady=2)
        tk.Button(left_panel, text='🔽 Click BOTTOM Point', command=self.start_calib_bottom,
                 bg='#FF9800', fg='white', font=('Arial', 10, 'bold'), width=25).pack(fill='x', padx=5, pady=2)
        
        self.label_calib = tk.Label(left_panel, text='🔴 Not calibrated', 
                                   font=('Arial', 9), bg='#fff3cd', fg='#856404', 
                                   wraplength=300, justify='left')
        self.label_calib.pack(fill='x', padx=5, pady=5)
        
        # ===== STEP 4: Meniscus Middle Point =====
        self._sep(left_panel)
        self._create_step_header(left_panel, '4️⃣ MENISCUS MIDDLE POINT')
        tk.Button(left_panel, text='🎯 Click Meniscus Middle Point', command=self.start_meniscus_middle,
                  bg='#9C27B0', fg='white', font=('Arial', 10, 'bold'), width=25).pack(fill='x', padx=5, pady=6)
        tk.Label(left_panel, text='(The meniscus line will be forced through this point)', font=('Arial', 8),
                bg='#f5f5f5').pack(fill='x', padx=5, pady=1)
        
        self.label_meniscus = tk.Label(left_panel, text='🔴 Meniscus middle not set',
                                      font=('Arial', 9), bg='#fff3cd', fg='#856404', 
                                      wraplength=300, justify='left')
        self.label_meniscus.pack(fill='x', padx=5, pady=5)

        # ===== STEP 5: Compute Water Height =====
        self._sep(left_panel)
        self._create_step_header(left_panel, '5️⃣ COMPUTE WATER HEIGHT')
        tk.Button(left_panel, text='▶ COMPUTE HEIGHT', command=self.run_detection,
                  bg='#f44336', fg='white', font=('Arial', 11, 'bold'),
                  width=25).pack(fill='x', padx=5, pady=6)

        self.label_result = tk.Label(left_panel, text='Ready...', 
                                    font=('Arial', 9), bg='#e8f5e9', fg='#1b5e20', 
                                    wraplength=300, justify='left')
        self.label_result.pack(fill='both', expand=True, padx=5, pady=5)
        
        # ===== UTILITIES =====
        self._sep(left_panel)
        frame_util = tk.Frame(left_panel, bg='#f5f5f5')
        frame_util.pack(fill='x', padx=5, pady=5)
        tk.Button(frame_util, text='💾 Save', command=self.save_setup, width=8).pack(side='left', padx=2)
        tk.Button(frame_util, text='📂 Load', command=self.load_setup, width=8).pack(side='left', padx=2)
        tk.Button(frame_util, text='🗑️ Reset', command=self.reset_all, width=8).pack(side='left', padx=2)
        
        # ============= RIGHT PANEL: CANVAS =============
        canvas_frame = tk.Frame(self, bg='black')
        canvas_frame.pack(side='right', fill='both', expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='black', cursor='crosshair')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Button-1>', self._on_canvas_click)
        self.canvas.bind('<B1-Motion>', self._on_canvas_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_canvas_release)
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        
        # Startup
        self.lift()
        self.attributes('-topmost', True)
        self.after(500, lambda: self.attributes('-topmost', False))
        logging.info('Manual Control GUI initialized')

    # ============= HELPER UI =============
    def _sep(self, parent):
        """Add visual separator"""
        tk.Frame(parent, height=2, bg='#ccc').pack(fill='x', pady=8)

    def _create_step_header(self, parent, text):
        """Create step header with styling"""
        tk.Label(parent, text=text, font=('Arial', 10, 'bold'), 
                bg='#f5f5f5', fg='#333').pack(fill='x', padx=5, pady=(5, 3))

    def _update_status(self):
        """Update all status labels"""
        # Image
        if self.current_image is not None:
            h, w = self.current_image.shape[:2]
            self.label_image.config(text=f'✅ Loaded ({w}×{h}px)\nIndex: {self.current_index + 1}/{len(self.image_paths)}',
                                   bg='#d4edda', fg='#155724')
        else:
            self.label_image.config(text='🔴 No image loaded', bg='#fff3cd', fg='#856404')
        
        # ROI
        if self.roi_rect:
            x1, y1, x2, y2 = self.roi_rect
            w, h = abs(x2-x1), abs(y2-y1)
            self.label_roi.config(text=f'✅ Rectangle:\n  ({x1}, {y1}) → ({x2}, {y2})\n  Size: {w}×{h}px',
                                 bg='#d4edda', fg='#155724')
        else:
            self.label_roi.config(text='🔴 No rectangle selected', bg='#fff3cd', fg='#856404')
        
        # Calibration
        if self.calib_top_point and self.calib_bottom_point:
            tx, ty = self.calib_top_point
            bx, by = self.calib_bottom_point
            dist_px = abs(by - ty)
            self.is_calibrated = True
            self.label_calib.config(text=f'✅ Calibrated:\n  Top: ({tx}, {ty})\n  Bottom: ({bx}, {by})\n  Distance: {dist_px}px = {self.mm_distance:.1f}mm',
                                   bg='#d4edda', fg='#155724')
        else:
            self.is_calibrated = False
            if self.calib_top_point:
                x, y = self.calib_top_point
                self.label_calib.config(text=f'🟡 Top set: ({x}, {y})\n  Need: Bottom point',
                                       bg='#fff3cd', fg='#856404')
            elif self.calib_bottom_point:
                x, y = self.calib_bottom_point
                self.label_calib.config(text=f'🟡 Bottom set: ({x}, {y})\n  Need: Top point',
                                       bg='#fff3cd', fg='#856404')
            else:
                self.label_calib.config(text='🔴 Not calibrated', bg='#fff3cd', fg='#856404')
        
        # Meniscus
        if self.meniscus_middle_point is not None:
            mx, my = self.meniscus_middle_point
            self.label_meniscus.config(text=f'✅ Meniscus middle set:\n  ({mx}, {my})',
                                      bg='#d4edda', fg='#155724')
        else:
            self.label_meniscus.config(text='🔴 Meniscus middle not set', bg='#fff3cd', fg='#856404')

    # ============= IMAGE LOADING =============
    def load_image(self):
        """Load single image"""
        p = filedialog.askopenfilename(
            filetypes=[('Images', '*.png;*.jpg;*.jpeg;*.tif;*.bmp')])
        if not p:
            return
        self.image_paths = [p]
        self.current_index = 0
        self._load_current_image()

    def load_folder(self):
        """Load all images from folder"""
        d = filedialog.askdirectory()
        if not d:
            return
        patterns = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
        paths = []
        for pattern in patterns:
            import glob
            paths.extend(glob.glob(os.path.join(d, pattern)))
        if not paths:
            messagebox.showerror('Error', 'No images found')
            return
        self.image_paths = sorted(paths)
        self.current_index = 0
        self._load_current_image()

    def prev_image(self):
        """Previous image in folder"""
        if not self.image_paths or len(self.image_paths) < 2:
            return
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self._load_current_image()

    def next_image(self):
        """Next image in folder"""
        if not self.image_paths or len(self.image_paths) < 2:
            return
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self._load_current_image()

    def _load_current_image(self):
        """Load and display current image"""
        if not self.image_paths:
            return
        path = self.image_paths[self.current_index]
        self.original_image = cv2.imread(path)
        if self.original_image is None:
            messagebox.showerror('Error', f'Failed to load {path}')
            return
        self.current_image = self.original_image.copy()
        self.meniscus_middle_point = None
        self.detected_meniscus_y = None
        self._show_image()
        self._update_status()

    # ============= DRAWING: ROI =============
    def start_draw_roi(self):
        """Start rectangle ROI drawing"""
        if self.current_image is None:
            messagebox.showerror('Error', 'Load image first')
            return
        self._drawing_roi = True
        messagebox.showinfo('Draw ROI', 'Click and drag on image to draw rectangle')

    def _on_canvas_click(self, evt):
        """Canvas mouse down"""
        if self.current_image is None:
            return
        img_x, img_y = self._canvas_to_image_coords(evt.x, evt.y)
        if img_x is None:
            return
        
        if self._drawing_roi:
            self._drag_start = (img_x, img_y)
        elif self._drawing_calib:
            if self._drawing_calib == 'top':
                self.calib_top_point = (img_x, img_y)
                self._drawing_calib = False
                logging.info(f'Set top point: {self.calib_top_point}')
                messagebox.showinfo('Set', f'Top point: {self.calib_top_point}')
            elif self._drawing_calib == 'bottom':
                self.calib_bottom_point = (img_x, img_y)
                self._drawing_calib = False
                logging.info(f'Set bottom point: {self.calib_bottom_point}')
                messagebox.showinfo('Set', f'Bottom point: {self.calib_bottom_point}')
            self.mm_distance = self.mm_var.get()
            self._update_status()
        elif self._drawing_meniscus_middle:
            self.meniscus_middle_point = (img_x, img_y)
            self.detected_meniscus_y = int(img_y)
            self._drawing_meniscus_middle = False
            logging.info(f'Set meniscus middle point: {self.meniscus_middle_point}')
            messagebox.showinfo('Set', f'Meniscus middle point: {self.meniscus_middle_point}')
            self._update_status()
    def _on_canvas_drag(self, evt):
        """Canvas mouse drag (for ROI rectangle)"""
        if not self._drawing_roi or not self._drag_start or self.current_image is None:
            return
        img_x, img_y = self._canvas_to_image_coords(evt.x, evt.y)
        if img_x is None:
            return
        
        # Draw preview rectangle
        x1, y1 = self._drag_start
        x2, y2 = img_x, img_y
        
        # Redraw canvas with preview
        self._show_image()
        
        # Draw rectangle preview on canvas
        c_x1, c_y1 = self._image_to_canvas_coords(x1, y1)
        c_x2, c_y2 = self._image_to_canvas_coords(x2, y2)
        if c_x1 is not None:
            if self._rect_canvas_id:
                self.canvas.delete(self._rect_canvas_id)
            self._rect_canvas_id = self.canvas.create_rectangle(
                c_x1, c_y1, c_x2, c_y2, outline='red', width=2)

    def _on_canvas_release(self, evt):
        """Canvas mouse up"""
        if self._drawing_roi and self._drag_start and self.current_image is not None:
            img_x, img_y = self._canvas_to_image_coords(evt.x, evt.y)
            if img_x is not None:
                x1, y1 = self._drag_start
                x2, y2 = img_x, img_y
                
                # Normalize rectangle
                self.roi_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                self._drawing_roi = False
                self._drag_start = None
                
                logging.info(f'ROI set: {self.roi_rect}')
                self._show_image()
                self._update_status()

    # ============= DRAWING: CALIBRATION & MENISCUS =============
    def start_calib_top(self):
        """Start selecting top calibration point"""
        if self.current_image is None:
            messagebox.showerror('Error', 'Load image first')
            return
        self._drawing_calib = 'top'
        messagebox.showinfo('Click Top', 'Click on the TOP point in image')

    def start_calib_bottom(self):
        """Start selecting bottom calibration point"""
        if self.current_image is None:
            messagebox.showerror('Error', 'Load image first')
            return
        self._drawing_calib = 'bottom'
        messagebox.showinfo('Click Bottom', 'Click on the BOTTOM point in image')

    def start_meniscus_middle(self):
        """Start selecting meniscus middle point (the meniscus line will be forced through it)."""
        if self.current_image is None:
            messagebox.showerror('Error', 'Load image first')
            return
        self._drawing_meniscus_middle = True
        messagebox.showinfo('Click Meniscus Middle', 'Click on the MENISCUS MIDDLE point in image')

    # ============= CANVAS COORDINATE CONVERSION =============
    def _on_canvas_resize(self, evt):
        """Canvas resize - redraw image"""
        if self.current_image is not None:
            self._show_image()

    def _show_image(self):
        """Display image on canvas with overlays"""
        if self.current_image is None:
            self.canvas.create_text(self.canvas.winfo_width()//2, 
                                   self.canvas.winfo_height()//2,
                                   text='No image', fill='white', font=('Arial', 16))
            return
        
        # Get canvas size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return
        
        # Convert image to PIL
        img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Scale to fit canvas
        img_h, img_w = self.current_image.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Store scale for coordinate conversion
        self._canvas_scale = scale
        self._canvas_offset_x = (canvas_w - new_w) / 2
        self._canvas_offset_y = (canvas_h - new_h) / 2
        
        # Convert to PhotoImage
        self.tk_image = ImageTk.PhotoImage(img_pil)
        
        # Clear and display
        self.canvas.delete('all')
        self.canvas.create_image(self._canvas_offset_x, self._canvas_offset_y, 
                                image=self.tk_image, anchor='nw')
        
        # Draw overlays
        self._draw_overlays()

    def _draw_overlays(self):
        """Draw ROI, calibration, meniscus points on canvas"""
        if self.current_image is None:
            return
        
        # ROI rectangle
        if self.roi_rect:
            x1, y1, x2, y2 = self.roi_rect
            c_x1, c_y1 = self._image_to_canvas_coords(x1, y1)
            c_x2, c_y2 = self._image_to_canvas_coords(x2, y2)
            if c_x1 is not None:
                self.canvas.create_rectangle(c_x1, c_y1, c_x2, c_y2, 
                                           outline='lime', width=2)
                self.canvas.create_text(c_x1+5, c_y1+5, text='ROI', 
                                       fill='lime', font=('Arial', 10, 'bold'), anchor='nw')
        
        # Top point
        if self.calib_top_point:
            x, y = self.calib_top_point
            c_x, c_y = self._image_to_canvas_coords(x, y)
            if c_x is not None:
                self.canvas.create_oval(c_x-5, c_y-5, c_x+5, c_y+5, 
                                       fill='yellow', outline='yellow', width=2)
                self.canvas.create_text(c_x+8, c_y-8, text='TOP', 
                                       fill='yellow', font=('Arial', 9, 'bold'), anchor='nw')
        
        # Bottom point
        if self.calib_bottom_point:
            x, y = self.calib_bottom_point
            c_x, c_y = self._image_to_canvas_coords(x, y)
            if c_x is not None:
                self.canvas.create_oval(c_x-5, c_y-5, c_x+5, c_y+5, 
                                       fill='cyan', outline='cyan', width=2)
                self.canvas.create_text(c_x+8, c_y+8, text='BOTTOM', 
                                       fill='cyan', font=('Arial', 9, 'bold'), anchor='nw')
                
                # Draw calibration line
                if self.calib_top_point:
                    t_x, t_y = self.calib_top_point
                    c_t_x, c_t_y = self._image_to_canvas_coords(t_x, t_y)
                    if c_t_x is not None:
                        self.canvas.create_line(c_t_x, c_t_y, c_x, c_y, 
                                              fill='orange', width=2, dash=(4, 4))
        
        # Detected meniscus line
        if self.detected_meniscus_y is not None:
            _, c_y = self._image_to_canvas_coords(0, self.detected_meniscus_y)
            if c_y is not None:
                left_x, _ = self._image_to_canvas_coords(0, self.detected_meniscus_y)
                right_x, _ = self._image_to_canvas_coords(self.current_image.shape[1] - 1, self.detected_meniscus_y)
                self.canvas.create_line(left_x, c_y, right_x, c_y, fill='magenta', width=3)
                self.canvas.create_text(left_x + 8, c_y - 8, text='MENISCUS',
                                       fill='magenta', font=('Arial', 9, 'bold'), anchor='nw')

        # Meniscus middle point marker (x,y)
        if self.meniscus_middle_point is not None:
            mx, my = self.meniscus_middle_point
            c_x, c_y = self._image_to_canvas_coords(mx, my)
            if c_x is not None and c_y is not None:
                self.canvas.create_oval(c_x - 6, c_y - 6, c_x + 6, c_y + 6,
                                        fill='magenta', outline='magenta', width=2)

    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas pixel to image pixel"""
        if not hasattr(self, '_canvas_scale'):
            return None, None
        img_x = (canvas_x - self._canvas_offset_x) / self._canvas_scale
        img_y = (canvas_y - self._canvas_offset_y) / self._canvas_scale
        
        # Bounds check
        h, w = self.current_image.shape[:2]
        if img_x < 0 or img_x >= w or img_y < 0 or img_y >= h:
            return None, None
        
        return int(img_x), int(img_y)

    def _image_to_canvas_coords(self, img_x, img_y):
        """Convert image pixel to canvas pixel"""
        if not hasattr(self, '_canvas_scale'):
            return None, None
        canvas_x = img_x * self._canvas_scale + self._canvas_offset_x
        canvas_y = img_y * self._canvas_scale + self._canvas_offset_y
        return canvas_x, canvas_y

    # ============= DETECTION =============
    def run_detection(self):
        """Compute water column height using the clicked meniscus middle point."""
        if self.current_image is None:
            messagebox.showerror('Error', 'Load image first')
            return
        if not self.roi_rect:
            messagebox.showerror('Error', 'Draw ROI first')
            return
        if not (self.calib_top_point and self.calib_bottom_point):
            messagebox.showerror('Error', 'Set top and bottom calibration points first')
            return
        if self.meniscus_middle_point is None:
            messagebox.showerror('Error', 'Set meniscus middle point first')
            return

        os.makedirs('results', exist_ok=True)

        if self.image_paths:
            base = os.path.splitext(os.path.basename(self.image_paths[self.current_index]))[0]
        else:
            base = 'result'
        out_path = os.path.join('results', f'{base}_manual_middle.png')

        try:
            self.mm_distance = self.mm_var.get()
            x1, y1, x2, y2 = self.roi_rect

            top_y = min(self.calib_top_point[1], self.calib_bottom_point[1])
            bottom_y = max(self.calib_top_point[1], self.calib_bottom_point[1])
            pixel_dist = abs(bottom_y - top_y)
            if pixel_dist == 0:
                raise ValueError('Calibration points must not be on the same row')

            scale_mm_per_pixel = float(self.mm_distance) / float(pixel_dist)

            mx, my = self.meniscus_middle_point
            meniscus_y = int(my)

            height_px = max(0, bottom_y - meniscus_y)
            water_height_mm = float(height_px) * float(scale_mm_per_pixel)

            annotated = self.original_image.copy()

            # ROI
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calibration points
            tx, ty = self.calib_top_point
            bx, by = self.calib_bottom_point
            cv2.circle(annotated, (tx, ty), 6, (0, 255, 255), -1)
            cv2.circle(annotated, (bx, by), 6, (255, 255, 0), -1)
            # Calibration line
            cv2.line(annotated, (0, top_y), (annotated.shape[1] - 1, top_y), (0, 200, 255), 2)
            cv2.line(annotated, (0, bottom_y), (annotated.shape[1] - 1, bottom_y), (0, 200, 255), 2)

            # Meniscus line forced through middle point
            cv2.line(annotated, (0, meniscus_y), (annotated.shape[1] - 1, meniscus_y), (255, 0, 255), 3)

            # Text
            text = f'Water Height: {water_height_mm:.2f} mm'
            cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imwrite(out_path, annotated)
            self.current_image = annotated
            self.detected_meniscus_y = meniscus_y

            self._show_image()
            self._update_status()

            result_text = '✅ Height Computed:\n'
            result_text += f'  Meniscus Y: {meniscus_y}px\n'
            result_text += f'  Water Height: {water_height_mm:.2f} mm\n'
            result_text += f'  Output: {out_path}'
            self.label_result.config(text=result_text, bg='#d4edda', fg='#1b5e20')

            logging.info(f'Height computed: {out_path}, water_height_mm={water_height_mm}')
            messagebox.showinfo('Done', f'Meniscus Y={meniscus_y}px\nWater height={water_height_mm:.2f}mm\n\nSaved to {out_path}')

        except Exception as e:
            messagebox.showerror('Error', f'Detection failed: {e}')
            logging.error(f'Detection error: {e}', exc_info=True)

    # ============= UTILITIES =============
    def save_setup(self):
        """Save current setup to JSON"""
        if self.current_image is None:
            messagebox.showerror('Error', 'Load image first')
            return
        
        setup = {
            'roi_rect': self.roi_rect,
            'calib_top_point': self.calib_top_point,
            'calib_bottom_point': self.calib_bottom_point,
            'meniscus_middle_point': self.meniscus_middle_point,
            'detected_meniscus_y': self.detected_meniscus_y,
            'mm_distance': self.mm_distance,
        }
        
        path = filedialog.asksaveasfilename(defaultextension='.json',
                                           filetypes=[('JSON', '*.json')])
        if not path:
            return
        
        try:
            with open(path, 'w') as f:
                json.dump(setup, f, indent=2)
            messagebox.showinfo('Saved', f'Setup saved to {path}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to save: {e}')

    def load_setup(self):
        """Load setup from JSON"""
        path = filedialog.askopenfilename(filetypes=[('JSON', '*.json')])
        if not path:
            return
        
        try:
            with open(path, 'r') as f:
                setup = json.load(f)
            
            self.roi_rect = tuple(setup.get('roi_rect')) if setup.get('roi_rect') else None
            self.calib_top_point = tuple(setup.get('calib_top_point')) if setup.get('calib_top_point') else None
            self.calib_bottom_point = tuple(setup.get('calib_bottom_point')) if setup.get('calib_bottom_point') else None
            self.meniscus_middle_point = tuple(setup.get('meniscus_middle_point')) if setup.get('meniscus_middle_point') else None
            self.detected_meniscus_y = setup.get('detected_meniscus_y')
            if self.detected_meniscus_y is None and self.meniscus_middle_point is not None:
                self.detected_meniscus_y = int(self.meniscus_middle_point[1])
            self.mm_distance = setup.get('mm_distance', 92.0)
            self.mm_var.set(self.mm_distance)
            
            self._show_image()
            self._update_status()
            messagebox.showinfo('Loaded', f'Setup loaded from {path}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load: {e}')

    def reset_all(self):
        """Reset all selections"""
        self.roi_rect = None
        self.calib_top_point = None
        self.calib_bottom_point = None
        self.meniscus_middle_point = None
        self.detected_meniscus_y = None
        self.mm_distance = 92.0
        self.mm_var.set(92.0)
        self._drawing_roi = False
        self._drawing_calib = False
        self._drawing_meniscus_middle = False
        self._drag_start = None

        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self._show_image()
        self._update_status()
        logging.info('Reset all selections')


if __name__ == '__main__':
    app = ManualControlGUI()
    app.mainloop()
