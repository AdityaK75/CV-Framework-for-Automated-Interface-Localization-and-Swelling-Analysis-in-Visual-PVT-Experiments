import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import json
import os
import logging
import cv2
import numpy as np
from pvt_analyzer import PVTAnalyzer


class MainGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('PVT Meniscus GUI')
        self.geometry('1200x800')

        self.analyzer = PVTAnalyzer()
        self.image_paths = []
        self.current_index = 0
        self.current_image = None  # original BGR image
        self.display_image = None  # PIL Image for display
        self.tk_image = None
        self.roi_box = None  # (x1,y1,x2,y2) in original image coords
        self.polygon_points = []
        self.calib_top = None
        self.calib_bottom = None
        self.middle_point = None

        # Left controls
        ctrl = tk.Frame(self, width=260)
        ctrl.pack(side='left', fill='y')

        tk.Button(ctrl, text='Open Image...', command=self.open_image).pack(fill='x')
        tk.Button(ctrl, text='Open Folder...', command=self.open_folder).pack(fill='x')
        tk.Button(ctrl, text='Prev', command=self.prev_image).pack(fill='x')
        tk.Button(ctrl, text='Next', command=self.next_image).pack(fill='x')

        tk.Label(ctrl, text='ROI Tools').pack()
        tk.Button(ctrl, text='Draw Rectangle ROI', command=self.start_rect).pack(fill='x')
        tk.Button(ctrl, text='Draw Polygon ROI', command=self.start_polygon).pack(fill='x')
        tk.Button(ctrl, text='Save ROI', command=self.save_roi).pack(fill='x')
        tk.Button(ctrl, text='Load ROI', command=self.load_roi).pack(fill='x')
        tk.Button(ctrl, text='Apply ROI', command=self.apply_roi).pack(fill='x')
        tk.Button(ctrl, text='Clear ROI', command=self.clear_roi).pack(fill='x')

        tk.Label(ctrl, text='Calibration (mm)').pack()
        self.mm_var = tk.DoubleVar(value=92.0)
        tk.Entry(ctrl, textvariable=self.mm_var).pack(fill='x')
        tk.Button(ctrl, text='Click Top', command=self.set_calib_top).pack(fill='x')
        tk.Button(ctrl, text='Click Bottom', command=self.set_calib_bottom).pack(fill='x')
        tk.Button(ctrl, text='Set Middle (Meniscus)', command=self.set_middle_point).pack(fill='x')
        tk.Button(ctrl, text='Apply Calibration', command=self.apply_calibration).pack(fill='x')

        tk.Label(ctrl, text='Detection').pack()
        self.method_var = tk.StringVar(value='basic')
        tk.OptionMenu(ctrl, self.method_var, 'basic', 'advanced', 'spanning', 'literature').pack(fill='x')
        self.contour_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text='Extract Contour', variable=self.contour_var).pack()
        tk.Button(ctrl, text='Run on Current', command=self.run_current).pack(fill='x')
        tk.Button(ctrl, text='Run Batch', command=self.run_batch).pack(fill='x')

        # Canvas for image
        self.canvas = tk.Canvas(self, bg='black')
        self.canvas.pack(side='right', fill='both', expand=True)
        self.canvas.bind('<Configure>', self._on_resize)
        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)

        # state for drawing
        self._drag_start = None
        self._rect_id = None
        self._poly_ids = []

        # Startup: log, raise window to front, and ensure it's visible
        try:
            logging.info('PVT Meniscus GUI started')
            # On macOS, explicitly bring window to foreground
            self.lift()
            self.attributes('-topmost', True)
            self.after(500, lambda: self.attributes('-topmost', False))  # revert after 500ms
        except Exception:
            # If window attributes fail, continue silently
            pass

    def open_image(self):
        p = filedialog.askopenfilename(filetypes=[('Images', '*.png;*.jpg;*.jpeg;*.tif;*.bmp')])
        if not p:
            return
        self.image_paths = [p]
        self.current_index = 0
        self._load_current()

    def open_folder(self):
        d = filedialog.askdirectory()
        if not d:
            return
        patterns = ['*.png','*.jpg','*.jpeg','*.tif','*.tiff','*.bmp']
        paths = []
        for pat in patterns:
            paths.extend(sorted([os.path.join(d,f) for f in os.listdir(d) if f.lower().endswith(pat.split('*')[-1])]))
        # fallback simple filter
        paths = [os.path.join(d,f) for f in os.listdir(d) if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff','.bmp'))]
        paths.sort()
        self.image_paths = paths
        self.current_index = 0
        if self.image_paths:
            self._load_current()

    def _load_current(self):
        path = self.image_paths[self.current_index]
        bgr = cv2.imread(path)
        if bgr is None:
            messagebox.showerror('Error', f'Could not load {path}')
            return
        self.current_image = bgr
        self.polygon_points = []
        self.roi_box = None
        self._show_image()

    def _show_image(self):
        if self.current_image is None:
            return
        h, w = self.current_image.shape[:2]
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 600
        scale = min(cw / w, ch / h)
        display = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        display = cv2.resize(display, (int(w*scale), int(h*scale)))
        self.display_image = Image.fromarray(display)
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.delete('all')
        self.canvas.create_image(0,0, anchor='nw', image=self.tk_image)
        # redraw ROI overlays if present
        if self.roi_box:
            x1,y1,x2,y2 = self._roi_to_canvas(self.roi_box)
            self._rect_id = self.canvas.create_rectangle(x1,y1,x2,y2, outline='lime', width=2)
        if self.polygon_points:
            self._draw_polygon()
        # draw calibration lines
        if self.analyzer.calib_lines:
            top,bottom = self.analyzer.calib_lines
            try:
                self._draw_calib_lines(top,bottom)
            except Exception:
                # safe fallback if method missing
                top_c = self._image_to_canvas((0, top))[1]
                bottom_c = self._image_to_canvas((0, bottom))[1]
                self.canvas.create_line(0, top_c, self.canvas.winfo_width(), top_c, fill='cyan', width=2)
                self.canvas.create_line(0, bottom_c, self.canvas.winfo_width(), bottom_c, fill='cyan', width=2)
        # draw middle point if set
        if self.middle_point is not None:
            mx, my = self.middle_point
            mx_c, my_c = self._image_to_canvas((mx, my))
            self.canvas.create_oval(mx_c-5, my_c-5, mx_c+5, my_c+5, outline='orange', width=3)
            self.canvas.create_line(0, my_c, self.canvas.winfo_width(), my_c, fill='orange', width=2)

    def _on_resize(self, event):
        self._show_image()

    def prev_image(self):
        if not self.image_paths:
            return
        self.current_index = max(0, self.current_index-1)
        self._load_current()

    def next_image(self):
        if not self.image_paths:
            return
        self.current_index = min(len(self.image_paths)-1, self.current_index+1)
        self._load_current()

    # --- ROI drawing helpers ---
    def start_rect(self):
        self._drag_start = None
        messagebox.showinfo('Rectangle ROI', 'Drag on the image to draw a rectangle ROI (click-drag-release)')

    def start_polygon(self):
        self.polygon_points = []
        self._poly_ids = []
        messagebox.showinfo('Polygon ROI', 'Click on the image to add polygon points. Right-click to close polygon.')

    def clear_roi(self):
        self.polygon_points = []
        self.roi_box = None
        self.analyzer.clear_manual_outline()
        self._show_image()

    def _on_click(self, event):
        if self.polygon_points is not None and len(self.polygon_points) >= 0 and self._rect_id is None and self._drag_start is None:
            # treat as polygon click if start_polygon was called recently - simple heuristic
            if self._poly_ids is not None and (len(self.polygon_points) > 0 or messagebox.askyesno('Polygon', 'Add polygon point at click?')):
                cx, cy = event.x, event.y
                ox, oy = self._canvas_to_image((cx, cy))
                self.polygon_points.append((ox, oy))
                self._draw_polygon()
                return
        # start rectangle drag
        self._drag_start = (event.x, event.y)

    def _on_drag(self, event):
        if self._drag_start:
            x0,y0 = self._drag_start
            x1,y1 = event.x, event.y
            # draw temporary rect
            if self._rect_id:
                self.canvas.delete(self._rect_id)
            self._rect_id = self.canvas.create_rectangle(x0,y0,x1,y1,outline='yellow',width=2)

    def _on_release(self, event):
        if self._drag_start:
            x0,y0 = self._drag_start
            x1,y1 = event.x, event.y
            ix0,iy0 = self._canvas_to_image((x0,y0))
            ix1,iy1 = self._canvas_to_image((x1,y1))
            xlo, xhi = sorted([ix0, ix1])
            ylo, yhi = sorted([iy0, iy1])
            self.roi_box = (xlo, ylo, xhi, yhi)
            # store rectangle ROI locally; applying to analyzer is manual via 'Apply ROI'
            # (do not auto-apply to analyzer to keep workflow fully manual)
            self._drag_start = None
            self._show_image()

    def _draw_polygon(self):
        # remove old
        for pid in self._poly_ids:
            try:
                self.canvas.delete(pid)
            except Exception:
                pass
        self._poly_ids = []
        if not self.polygon_points:
            return
        # draw points and lines
        pts_canvas = [self._image_to_canvas(pt) for pt in self.polygon_points]
        for (x,y) in pts_canvas:
            self._poly_ids.append(self.canvas.create_oval(x-3,y-3,x+3,y+3, fill='red'))
        if len(pts_canvas) > 1:
            flat = []
            for x,y in pts_canvas:
                flat.extend([x,y])
            self._poly_ids.append(self.canvas.create_polygon(flat, outline='lime', fill='', width=2))

    def _image_to_canvas(self, pt):
        # convert original image coords to canvas coords based on current display scaling
        x,y = pt
        h,w = self.current_image.shape[:2]
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 600
        scale = min(cw / w, ch / h)
        return int(x*scale), int(y*scale)

    def _canvas_to_image(self, pt):
        x,y = pt
        h,w = self.current_image.shape[:2]
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 600
        scale = min(cw / w, ch / h)
        return int(x/scale), int(y/scale)

    def _roi_to_canvas(self, roi):
        x1,y1,x2,y2 = roi
        x1c,y1c = self._image_to_canvas((x1,y1))
        x2c,y2c = self._image_to_canvas((x2,y2))
        return x1c,y1c,x2c,y2c

    def _draw_calib_lines(self, top, bottom):
        # draw horizontal calibration lines (top and bottom) on canvas
        top_c = self._image_to_canvas((0, top))[1]
        bottom_c = self._image_to_canvas((0, bottom))[1]
        self.canvas.create_line(0, top_c, self.canvas.winfo_width(), top_c, fill='cyan', width=2)
        self.canvas.create_line(0, bottom_c, self.canvas.winfo_width(), bottom_c, fill='cyan', width=2)

    def set_middle_point(self):
        if self.current_image is None:
            messagebox.showerror('Error','Load an image first')
            return
        messagebox.showinfo('Middle Point','Click a point on image to mark the MENISCUS (middle)')
        self.canvas.bind('<Button-1>', self._on_set_middle)

    def _on_set_middle(self, event):
        ox,oy = self._canvas_to_image((event.x,event.y))
        self.middle_point = (ox,oy)
        self.canvas.unbind('<Button-1>')
        self._show_image()
        messagebox.showinfo('Middle Point','Middle point recorded')

    def apply_roi(self):
        # Apply either polygon or rectangle ROI to the analyzer explicitly (manual action)
        if self.current_image is None:
            messagebox.showerror('Error','Load an image first')
            return
        if self.polygon_points and len(self.polygon_points) >= 3:
            try:
                self.analyzer.set_manual_outline(self.current_image, self.polygon_points)
                # update roi_box from analyzer.roi_coords if available
                if getattr(self.analyzer, 'roi_coords', None):
                    ry1, ry2, rx1, rx2 = self.analyzer.roi_coords
                    self.roi_box = (rx1, ry1, rx2, ry2)
            except Exception as e:
                messagebox.showerror('Error', f'Failed to set polygon ROI: {e}')
                return
        elif self.roi_box:
            x1,y1,x2,y2 = self.roi_box
            try:
                self.analyzer.set_roi(y1, y2, x1, x2)
            except Exception as e:
                messagebox.showerror('Error', f'Failed to set rectangle ROI: {e}')
                return
        else:
            messagebox.showerror('Error','No ROI drawn to apply')
            return
        self._show_image()
        messagebox.showinfo('ROI', 'ROI applied to analyzer')

    def set_calib_top(self):
        if self.current_image is None:
            messagebox.showerror('Error','Load an image first')
            return
        messagebox.showinfo('Calibration','Click a point on image to mark TOP calibration')
        self.canvas.bind('<Button-1>', self._on_set_top)

    def _on_set_top(self, event):
        ox,oy = self._canvas_to_image((event.x,event.y))
        self.calib_top = (ox,oy)
        self.canvas.unbind('<Button-1>')
        self._show_image()
        messagebox.showinfo('Calibration','Top point recorded')

    def set_calib_bottom(self):
        if self.current_image is None:
            messagebox.showerror('Error','Load an image first')
            return
        messagebox.showinfo('Calibration','Click a point on image to mark BOTTOM calibration')
        self.canvas.bind('<Button-1>', self._on_set_bottom)

    def _on_set_bottom(self, event):
        ox,oy = self._canvas_to_image((event.x,event.y))
        self.calib_bottom = (ox,oy)
        self.canvas.unbind('<Button-1>')
        self._show_image()
        messagebox.showinfo('Calibration','Bottom point recorded')

    def apply_calibration(self):
        if self.calib_top is None or self.calib_bottom is None:
            messagebox.showerror('Error','Set both top and bottom calibration points first')
            return
        dy = abs(self.calib_bottom[1] - self.calib_top[1])
        mm = float(self.mm_var.get() or 92.0)
        if dy <= 0:
            messagebox.showerror('Error','Calibration distance is zero')
            return
        scale = mm / dy
        self.analyzer.scale_mm_per_pixel = scale
        self.analyzer.calib_lines = (min(self.calib_top[1], self.calib_bottom[1]), max(self.calib_top[1], self.calib_bottom[1]))
        messagebox.showinfo('Calibration', f'Calibration set: {scale:.6f} mm/px')

    def save_roi(self):
        """Save current ROI, polygon, calibration and middle point to a JSON file."""
        if self.current_image is None:
            messagebox.showerror('Error','Load an image first')
            return
        data = {
            'roi_box': self.roi_box,
            'polygon_points': self.polygon_points,
            'calib_top': self.calib_top,
            'calib_bottom': self.calib_bottom,
            'mm': float(self.mm_var.get() or 92.0),
            'middle_point': self.middle_point,
        }
        # suggest file next to current image
        cur_path = self.image_paths[self.current_index] if self.image_paths else None
        initialfile = None
        if cur_path:
            base = os.path.splitext(os.path.basename(cur_path))[0]
            initialfile = base + '.roi.json'

        p = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('ROI JSON','*.json')], initialfile=initialfile)
        if not p:
            return
        try:
            # Convert any numpy types to python types
            def clean(obj):
                if obj is None:
                    return None
                if isinstance(obj, (list, tuple)):
                    return [clean(x) for x in obj]
                if isinstance(obj, (int, float, str)):
                    return obj
                return obj
            with open(p, 'w') as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo('Saved', f'ROI saved to {p}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to save ROI: {e}')

    def load_roi(self):
        """Load ROI JSON and apply to GUI + analyzer (manual apply semantics preserved)."""
        if self.current_image is None:
            messagebox.showerror('Error','Load an image first')
            return
        p = filedialog.askopenfilename(filetypes=[('ROI JSON','*.json')])
        if not p:
            return
        try:
            with open(p, 'r') as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to read ROI: {e}')
            return

        # Apply loaded fields to GUI state
        try:
            self.roi_box = tuple(data.get('roi_box')) if data.get('roi_box') else None
        except Exception:
            self.roi_box = None
        try:
            pts = data.get('polygon_points')
            if pts:
                # ensure list of tuples
                self.polygon_points = [tuple(p) for p in pts]
                # apply polygon to analyzer
                try:
                    self.analyzer.set_manual_outline(self.current_image, self.polygon_points)
                    if getattr(self.analyzer, 'roi_coords', None):
                        ry1, ry2, rx1, rx2 = self.analyzer.roi_coords
                        self.roi_box = (rx1, ry1, rx2, ry2)
                except Exception:
                    # keep polygon in GUI even if analyzer rejects
                    pass
        except Exception:
            self.polygon_points = []

        # If there was a rectangle ROI, set it in analyzer as well
        if self.roi_box:
            try:
                x1,y1,x2,y2 = self.roi_box
                # analyzer expects set_roi(y1,y2,x1,x2)
                self.analyzer.set_roi(y1, y2, x1, x2)
            except Exception:
                pass

        # calibration
        ct = data.get('calib_top')
        cb = data.get('calib_bottom')
        if ct:
            self.calib_top = tuple(ct)
        if cb:
            self.calib_bottom = tuple(cb)
        mm = data.get('mm')
        if mm:
            try:
                self.mm_var.set(float(mm))
            except Exception:
                pass

        # apply calibration if both points present
        if self.calib_top and self.calib_bottom:
            try:
                self.apply_calibration()
            except Exception:
                pass

        # middle point
        mp = data.get('middle_point')
        if mp:
            try:
                self.middle_point = tuple(mp)
            except Exception:
                self.middle_point = None

        self._show_image()
        messagebox.showinfo('Loaded', f'ROI loaded from {p}')

    def run_current(self):
        if not self.image_paths:
            messagebox.showerror('Error','No image loaded')
            return
        path = self.image_paths[self.current_index]
        out_dir = os.path.join(os.getcwd(),'results')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + '_detected.png')
        try:
            # If user explicitly set a middle point, use it as the meniscus Y
            if self.middle_point is not None:
                mx, my = self.middle_point
                # Ensure middle point is within ROI if ROI exists
                if getattr(self.analyzer, 'roi_coords', None):
                    ry1, ry2, rx1, rx2 = self.analyzer.roi_coords
                    if not (rx1 <= mx <= rx2 and ry1 <= my <= ry2):
                        if not messagebox.askyesno('Middle Point Outside ROI', 'Middle point is outside the current ROI. Use it anyway?'):
                            return
                # build annotated image using analyzer helpers
                annotated = cv2.imread(path)
                cv2.line(annotated, (0, my), (annotated.shape[1]-1, my), (0, 165, 255), 3)
                # draw ROI if available
                if getattr(self.analyzer, 'roi_coords', None):
                    ry1, ry2, rx1, rx2 = self.analyzer.roi_coords
                    cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
                # compute height in mm if calibrated
                height_mm = None
                if getattr(self.analyzer, 'scale_mm_per_pixel', None) is not None and getattr(self.analyzer, 'calib_lines', None) is not None:
                    top_y, bottom_y = self.analyzer.calib_lines
                    height_pixels = max(0, bottom_y - int(my))
                    height_mm = float(height_pixels) * float(self.analyzer.scale_mm_per_pixel)
                    text = f"Height: {height_mm:.2f} mm"
                    cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                # save annotated
                cv2.imwrite(out_path, annotated)
                messagebox.showinfo('Done', f"Applied middle point for {path}\nHeight(mm): {height_mm}")
                # reload annotated
                ann = cv2.imread(out_path)
                if ann is not None:
                    self.current_image = ann
                    self._show_image()
            else:
                res = self.analyzer.process_and_annotate(path, out_path=out_path, method=self.method_var.get(), visualize=False, auto_crop=False, verbose=True, extract_contour=self.contour_var.get())
                messagebox.showinfo('Done', f"Processed {path}\nHeight(mm): {res.get('height_mm')}")
                # reload annotated
                ann = cv2.imread(res['out_path'])
                if ann is not None:
                    self.current_image = ann
                    self._show_image()
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def run_batch(self):
        if not self.image_paths:
            messagebox.showerror('Error','No images loaded')
            return
        out_dir = os.path.join(os.getcwd(),'results')
        os.makedirs(out_dir, exist_ok=True)
        for p in self.image_paths:
            out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + '_detected.png')
            try:
                self.analyzer.process_and_annotate(p, out_path=out_path, method=self.method_var.get(), visualize=False, auto_crop=False, verbose=True, extract_contour=self.contour_var.get())
            except Exception as e:
                print('Failed', p, e)
        messagebox.showinfo('Done', 'Batch processing finished')


# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    # Create the application window
    app = MainGUI()
    
    # Start the Tkinter event loop (this keeps the window open)
    app.mainloop()

