# Standard library imports
import os
import json
import logging
import traceback
from datetime import datetime

# Computer vision & numerical processing
import cv2
import numpy as np
import pandas as pd

# Yolo
from ultralytics import YOLO

# Smoothing
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

# Visualization & plotting
import matplotlib.pyplot as plt
import seaborn as sns

# GUI
import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk             

# ----------------------------- CONFIG -----------------------------
MODEL_PATH = "models/YoloV8n_mouse.pt"    # Path to YOLOv8 model file
HEATMAP_SIGMA = 20                        # Gaussian sigma for heatmap smoothing
SMOOTH_WINDOW = 11                        # Savitzky-Golay window length
SMOOTH_POLYORDER = 3                      # Savitzky-Golay polynomial order
OUTPUT_DIR = "OFT_Results"                # Output folder
LOGFILE = os.path.join(OUTPUT_DIR, "oft_run.log")
# ------------------------------------------------------------------

# set GUI theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Configure logging
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOGFILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)


def ensure_odd_positive(n):
    """Ensure n is odd and >=3 for savgol; if not, adjust"""
    n = int(n)
    if n < 3:
        return 3
    if n % 2 == 0:
        return n + 1
    return n


class StartupWindow(ctk.CTk):
    """Opening window for selecting live camera or video file"""

    def __init__(self):
        super().__init__()
        self.title("Open Field Test - Choose Input")
        self.geometry("600x500")
        self.resizable(False, False)

        ctk.CTkLabel(self, text="Open Field Test Tracker", font=("Arial", 32, "bold")).pack(pady=50)
        ctk.CTkLabel(self, text="Select input source:", font=("Arial", 18)).pack(pady=20)

        ctk.CTkButton(self, text="Live Camera (Webcam / USB)", height=60, font=("Arial", 20),
                      command=lambda: self.start_mode("camera")).pack(pady=20, fill="x", padx=100)
        ctk.CTkButton(self, text="Video File", height=60, font=("Arial", 20),
                      command=lambda: self.start_mode("video")).pack(pady=10, fill="x", padx=100)

    def start_mode(self, mode):
        """Launch main application window with the chosen mode"""
        self.destroy()
        MainApp(mode).mainloop()


class MainApp(ctk.CTk):
    """Main GUI that sets up parameters and runs the analysis pipeline"""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.title("OFT Tracker - Running")
        self.geometry("980x780")

        # video capture & params
        self.cap = None
        self.video_path = None
        self.mouse_id_var = ctk.StringVar(value="Mouse001")

        # calibration: set after arena drawn and user input
        self.px_per_cm = None  # pixels per centimeter (if calibrated)

        # Load model once
        try:
            logging.info(f"Loading YOLO model from {MODEL_PATH}")
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            logging.exception("Failed to load YOLO model")
            messagebox.showerror("Model Error", f"Failed to load model: {e}")
            raise

        self.setup_ui()

        # Start input selection
        if mode == "camera":
            self.ask_camera_id()
        else:
            self.select_video_file()

    def setup_ui(self):
        """Construct the UI controls"""
        ctk.CTkLabel(self, text="Open Field Test Analysis", font=("Arial", 28, "bold")).pack(pady=14)

        frame = ctk.CTkFrame(self)
        frame.pack(pady=10, padx=60, fill="x")

        ctk.CTkLabel(frame, text="Mouse ID:", font=("Arial", 16)).pack(side="left", padx=12)
        ctk.CTkEntry(frame, textvariable=self.mouse_id_var, width=320, font=("Arial", 16)).pack(side="left", padx=8)

        # Calibration toggle (optional)
        self.calib_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(frame, text="Calibrate (px > cm)", variable=self.calib_var).pack(side="left", padx=20)

        self.status = ctk.CTkLabel(self, text="Ready", font=("Arial", 14))
        self.status.pack(pady=12)

        self.start_btn = ctk.CTkButton(self, text="Start Analysis", height=55, font=("Arial", 18, "bold"),
                                       command=self.start_processing)
        self.start_btn.pack(pady=18)

    # Input selection handlers
    def ask_camera_id(self):
        dialog = ctk.CTkInputDialog(text="Enter Camera ID (0 = built-in, 1,2,... = USB):", title="Camera ID")
        cam_id = dialog.get_input()
        try:
            cam_id = int(cam_id or 0)
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                messagebox.showerror("Error", "Cannot open camera")
                return
            self.cap = cap
            self.status.configure(text=f"Camera {cam_id} connected - Click Start")
        except Exception:
            messagebox.showerror("Error", "Invalid camera ID")
            self.destroy()

    def select_video_file(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if path:
            self.video_path = path
            self.status.configure(text=f"Video loaded: {os.path.basename(path)}\nClick Start Analysis")
        else:
            messagebox.showerror("Error", "No video selected")
            self.destroy()


    # Run pipeline
    def start_processing(self):
        """Validates input and runs the analysis pipeline."""
        mouse_id = self.mouse_id_var.get().strip() or "Unknown"

        if self.mode == "camera" and self.cap is None:
            messagebox.showerror("Error", "Camera not connected")
            return
        if self.mode == "video" and not self.video_path:
            messagebox.showerror("Error", "No video selected")
            return

        # disable UI controls while processing
        self.start_btn.configure(state="disabled")
        self.status.configure(text="Draw arena > Click points > Press ENTER")

        try:
            self.run_analysis(mouse_id, calibrate=self.calib_var.get())
            messagebox.showinfo("Success!", f"Analysis complete. Results in '{OUTPUT_DIR}'.")
        except Exception as e:
            logging.error("Processing failed:\n%s", traceback.format_exc())
            messagebox.showerror("Error", f"Processing failed: {e}")
        finally:
            self.start_btn.configure(state="normal")
            self.status.configure(text="Ready")

    # Arena drawing & calibration
    def draw_arena(self, frame):
        """
        Let the user click to define a polygon arena. Press ENTER to finish (>=3 points).
        Press 'r' to reset.
        Returns: list of (x,y) points.
        """
        points = []
        clone = frame.copy()

        def click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))

        winname = "Draw Arena - Click points > ENTER (r=reset)"
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(winname, click)

        while True:
            img = clone.copy()
            if points:
                cv2.polylines(img, [np.array(points)], False, (0, 255, 0), 2)
                for p in points:
                    cv2.circle(img, p, 6, (0, 255, 0), -1)
            cv2.imshow(winname, img)
            k = cv2.waitKey(20) & 0xFF
            if k == 13 and len(points) >= 3:  # ENTER
                break
            if k == ord('r'):
                points.clear()
        cv2.destroyWindow(winname)
        return points

    def calibrate_pixels_to_cm(self, x1, x2):
        """
        Ask user for real-world arena width (cm) and compute px_per_cm using two x coordinates.
        x1, x2: leftmost and rightmost pixel positions of arena bounding box.
        Returns px_per_cm (float) or None if user cancels.
        """
        dlg = ctk.CTkInputDialog(text="Enter arena width in cm (e.g. 40):", title="Calibration (cm)")
        val = dlg.get_input()
        try:
            if val is None or val.strip() == "":
                return None
            width_cm = float(val.strip())
            px_width = abs(x2 - x1)
            if width_cm <= 0 or px_width <= 0:
                return None
            return px_width / width_cm
        except Exception:
            messagebox.showerror("Calibration Error", "Invalid numeric value for cm.")
            return None

    # Analysis core
    def run_analysis(self, mouse_id, calibrate=False):
        """
        Main processing pipeline:
        - Open video/camera
        - Draw arena polygon
        - Optional calibration
        - Per-frame detection -> choose best box -> center
        - Smoothing, metrics, figures, save outputs
        """

        is_camera = self.mode == "camera"
        cap = self.cap if is_camera else cv2.VideoCapture(self.video_path)

        # Read basic video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info("Video properties: FPS=%.2f, WxH=%dx%d", fps, frame_w, frame_h)

        # Read first frame to draw arena
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Unable to read first frame from source.")

        # Let user draw polygon arena
        arena_poly = self.draw_arena(frame)
        arena_np = np.array(arena_poly, dtype=np.int32)

        # Compute axis-aligned bbox of arena for grid & center zones
        x_min, y_min = arena_np.min(axis=0).astype(int)
        x_max, y_max = arena_np.max(axis=0).astype(int)

        # Optional calibration: ask for arena width in cm -> px_per_cm
        px_per_cm = None
        if calibrate:
            px_per_cm = self.calibrate_pixels_to_cm(x_min, x_max)
            if px_per_cm:
                self.px_per_cm = px_per_cm
                logging.info("Calibration: %.4f px per cm", px_per_cm)
            else:
                logging.info("Calibration not provided or canceled. Metrics will remain in pixels.")

        # Prepare outputs
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base = os.path.join(OUTPUT_DIR, mouse_id)
        tracked_video_path = f"{base}_tracked.mp4"
        summary_xlsx = f"{base}_Summary.xlsx"
        perframe_csv = f"{base}_PerFrame.csv"
        heatmap_png = f"{base}_Heatmap.png"
        traj_png = f"{base}_Trajectory.png"
        speed_png = f"{base}_SpeedTimeSeries.png"
        metadata_json = f"{base}_metadata.json"

        # Video writer (match input frame size & fps)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(tracked_video_path, fourcc, fps, (frame_w, frame_h))

        # Data containers
        rows = []
        heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)
        traj = []
        prev = None
        total_px_dist = 0.0
        frame_idx = 0

        # Precompute center zones (4x4 grid, central 2x2 is "center")
        w = x_max - x_min
        h = y_max - y_min
        cell_w, cell_h = w // 4, h // 4
        center_zones = []
        center_centers = []
        for r in range(1, 3):
            for c in range(1, 3):
                zx1 = x_min + c * cell_w
                zy1 = y_min + r * cell_h
                center_zones.append((zx1, zy1, zx1 + cell_w, zy1 + cell_h))
                center_centers.append((zx1 + cell_w // 2, zy1 + cell_h // 2))

        # Reset to frame 0 if reading video
        if not is_camera:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Processing loop
        logging.info("Starting frame-by-frame tracking...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                # YOLO track — returns list-like results; pick first results
                res = self.model.track(frame, persist=True, conf=0.4, iou=0.5, verbose=False)[0]
            except Exception:
                # If model.track fails for frame, skip but log
                logging.warning("YOLO tracking failed on frame %d. Skipping frame.", frame_idx)
                frame_idx += 1
                continue

            ann = frame.copy()

            # Draw arena polygon / grid
            cv2.polylines(ann, [arena_np], isClosed=True, color=(255, 255, 255), thickness=2)
            # inner grid
            for i in range(1, 4):
                cv2.line(ann, (x_min + i * cell_w, y_min), (x_min + i * cell_w, y_max), (220, 220, 220), 1)
                cv2.line(ann, (x_min, y_min + i * cell_h), (x_max, y_min + i * cell_h), (220, 220, 220), 1)
            # center highlight
            overlay = ann.copy()
            for z in center_zones:
                cv2.rectangle(overlay, (z[0], z[1]), (z[2], z[3]), (255, 255, 0), -1)
            cv2.addWeighted(overlay, 0.18, ann, 0.82, 0, ann)
            for d in center_centers:
                cv2.circle(ann, d, 6, (0, 0, 255), -1)

            # Extract detections -> boxes
            cx, cy = None, None
            if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                # Choose best box: prefer the one with largest area (robust if multiple detections)
                boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") else np.array(res.boxes.xyxy)
                # boxes shape: (N, 4)
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best_idx = int(np.argmax(areas))
                xb1, yb1, xb2, yb2 = map(int, boxes[best_idx])
                cx = (xb1 + xb2) // 2
                cy = (yb1 + yb2) // 2

                # Draw bounding box & label
                cv2.rectangle(ann, (xb1, yb1), (xb2, yb2), (0, 200, 0), 2)
                cv2.putText(ann, mouse_id, (xb1, max(0, yb1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

            # If we have a center, compute metrics and append row
            if cx is not None and cy is not None:
                # Determine zone: center vs border
                in_center = any((cx >= z[0] and cx <= z[2] and cy >= z[1] and cy <= z[3]) for z in center_zones)
                zone = "Center" if in_center else "Border"

                # Distance & speed (pixels)
                if prev is not None:
                    d_px = float(np.hypot(cx - prev[0], cy - prev[1]))
                    total_px_dist += d_px
                    speed_px_s = d_px * fps  # instantaneous speed estimate (px per second)
                else:
                    d_px = 0.0
                    speed_px_s = 0.0

                prev = (cx, cy)

                # Record heatmap & trajectory
                # clamp to image bounds to avoid index errors
                ix = np.clip(int(round(cx)), 0, frame_w - 1)
                iy = np.clip(int(round(cy)), 0, frame_h - 1)
                heatmap[iy, ix] += 1.0
                traj.append((cx, cy))

                # store per-frame data
                rows.append({
                    "Frame": frame_idx,
                    "Time_s": round(frame_idx / fps, 4),
                    "X_px": cx,
                    "Y_px": cy,
                    "Zone": zone,
                    "Speed_px_s": round(speed_px_s, 3),
                    "Delta_px": round(d_px, 3),
                })

            # draw recent trajectory (last 200 points)
            if len(traj) > 1:
                pts = np.array(traj[-200:], dtype=np.int32)
                cv2.polylines(ann, [pts], False, (0, 0, 255), 2)

            # write annotated frame
            out_writer.write(ann)

            # show minimal live view for feedback
            cv2.imshow("OFT Live Tracking - Press 'q' to stop", ann)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("User requested stop (q pressed).")
                break

            frame_idx += 1

        # cleanup capture & windows
        cap.release()
        out_writer.release()
        cv2.destroyAllWindows()

        logging.info("Frames processed: %d", frame_idx)
        # Build DataFrame from rows
        df = pd.DataFrame(rows)

        # If no detections found -> raise
        if df.empty:
            raise RuntimeError("No detections found. Check model confidence or video quality.")

        # Apply smoothing to trajectories
        logging.info("Smoothing trajectory with Savitzky-Golay filter...")
        traj_arr = np.array([(r["X_px"], r["Y_px"]) for r in rows], dtype=float)
        n_points = traj_arr.shape[0]
        win = ensure_odd_positive(min(SMOOTH_WINDOW, n_points // 2 * 2 - 1 if n_points % 2 == 0 else min(SMOOTH_WINDOW, n_points)))
        if win < 3:
            win = 3
        win = ensure_odd_positive(win)
        try:
            xs_smooth = savgol_filter(traj_arr[:, 0], win, SMOOTH_POLYORDER)
            ys_smooth = savgol_filter(traj_arr[:, 1], win, SMOOTH_POLYORDER)
        except Exception:
            logging.warning("Savitzky-Golay smoothing failed; using unsmoothed trajectory.")
            xs_smooth = traj_arr[:, 0]
            ys_smooth = traj_arr[:, 1]

        df["X_px_smooth"] = xs_smooth
        df["Y_px_smooth"] = ys_smooth

        # Recompute smoothed speeds (cm or px)
        dx = np.diff(xs_smooth, prepend=xs_smooth[0])
        dy = np.diff(ys_smooth, prepend=ys_smooth[0])
        dist_smooth_px = np.hypot(dx, dy)
        df["Speed_px_s_smooth"] = dist_smooth_px * fps

        # Optionally convert to cm units if p`x_per_cm provided
        if self.px_per_cm:
            px2cm = 1.0 / self.px_per_cm
            df["X_cm"] = df["X_px_smooth"] * px2cm
            df["Y_cm"] = df["Y_px_smooth"] * px2cm
            df["Speed_cm_s"] = df["Speed_px_s_smooth"] * px2cm
            total_dist_cm = total_px_dist * px2cm
            units = "cm"
        else:
            total_dist_cm = None
            units = "pixels"
            df["Speed_cm_s"] = np.nan  # maintain column for consistency

        # Compute summary metrics
        total_time_s = df["Time_s"].max() - df["Time_s"].min() if df.shape[0] > 1 else 0.0
        mean_speed = float(df["Speed_px_s_smooth"].mean())
        median_speed = float(df["Speed_px_s_smooth"].median())
        max_speed = float(df["Speed_px_s_smooth"].max())
        path_length_px = float(dist_smooth_px.sum())
        path_length_cm = path_length_px * (1.0 / self.px_per_cm) if self.px_per_cm else None

        # Center occupancy & latency
        df["In_Center"] = df.apply(
            lambda r: any((r["X_px"] >= z[0] and r["X_px"] <= z[2] and r["Y_px"] >= z[1] and r["Y_px"] <= z[3]) for z in center_zones),
            axis=1
        )
        time_in_center_s = df["In_Center"].sum() / fps
        pct_time_center = (time_in_center_s / total_time_s * 100) if total_time_s else 0.0

        # Latency to first center entry
        try:
            first_center_idx = df.index[df["In_Center"]].tolist()[0]
            latency_to_center_s = df.loc[first_center_idx, "Time_s"]
        except IndexError:
            latency_to_center_s = None

        # Thigmotaxis: time near walls (within one cell width from edges)
        margin = int(min(cell_w, cell_h) * 0.5)  # e.g., half cell width
        near_wall_mask = (
            (df["X_px"] <= x_min + margin) |
            (df["X_px"] >= x_max - margin) |
            (df["Y_px"] <= y_min + margin) |
            (df["Y_px"] >= y_max - margin)
        )
        time_near_wall_s = near_wall_mask.sum() / fps
        pct_time_near_wall = (time_near_wall_s / total_time_s * 100) if total_time_s else 0.0

        # Center entries count (edge crossing)
        center_entries = ((df["In_Center"]) & (~df["In_Center"].shift(1).fillna(False))).sum()

        # Save outputs
        logging.info("Saving per-frame CSV to %s", perframe_csv)
        df.to_csv(perframe_csv, index=False)

        logging.info("Saving summary Excel to %s", summary_xlsx)
        summary = {
            "Mouse_ID": mouse_id,
            "Source": "Camera" if is_camera else "Video File",
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Total_Time_s": round(total_time_s, 3),
            "Path_Length_px": round(path_length_px, 3),
            "Path_Length_cm": round(path_length_cm, 3) if path_length_cm is not None else None,
            "Total_Distance_px_raw": round(total_px_dist, 3),
            "Mean_Speed_px_s": round(mean_speed, 3),
            "Median_Speed_px_s": round(median_speed, 3),
            "Max_Speed_px_s": round(max_speed, 3),
            "Time_in_Center_s": round(time_in_center_s, 3),
            "Pct_Time_in_Center": round(pct_time_center, 3),
            "Center_Entries": int(center_entries),
            "Latency_to_Center_s": round(latency_to_center_s, 3) if latency_to_center_s is not None else None,
            "Time_near_wall_s": round(time_near_wall_s, 3),
            "Pct_Time_near_wall": round(pct_time_near_wall, 3),
            "Units": units,
            "Px_per_cm": round(self.px_per_cm, 4) if self.px_per_cm else None,
            "Input_FPS": fps,
            "Frames_Processed": int(frame_idx),
        }

        # Write summary as Excel (sheet 1 = summary, sheet 2 = per-frame)
        with pd.ExcelWriter(summary_xlsx) as writer:
            pd.DataFrame.from_dict(summary, orient="index", columns=["Value"]).to_excel(writer, sheet_name="Summary", header=False)
            df.to_excel(writer, sheet_name="PerFrame", index=False)

        # write metadata json
        metadata = {
            "mouse_id": mouse_id,
            "model_path": MODEL_PATH,
            "timestamp": datetime.now().isoformat(),
            "video_props": {"width": frame_w, "height": frame_h, "fps": fps},
            "summary": summary
        }
        with open(metadata_json, "w") as f:
            json.dump(metadata, f, indent=2)

        # -----------------------------
        # Heatmap & figures
        # -----------------------------
        logging.info("Generating heatmap and figures...")

        # Heatmap (smoothed)
        plt.figure(figsize=(8, 6))
        hm = gaussian_filter(heatmap, HEATMAP_SIGMA)
        sns.heatmap(hm, cmap="hot", cbar_kws={"label": "Frame count"})
        plt.title(f"OFT Heatmap - {mouse_id}")
        plt.axis("off")
        plt.savefig(heatmap_png, dpi=300, bbox_inches="tight")
        plt.close()

        # Full trajectory image
        traj_img = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        # draw arena outline and grid
        cv2.polylines(traj_img, [arena_np], isClosed=True, color=(255, 255, 255), thickness=2)
        for i in range(1, 4):
            cv2.line(traj_img, (x_min + i * cell_w, y_min), (x_min + i * cell_w, y_max), (150, 150, 150), 1)
            cv2.line(traj_img, (x_min, y_min + i * cell_h), (x_max, y_min + i * cell_h), (150, 150, 150), 1)
        # center zones
        for z in center_zones:
            cv2.rectangle(traj_img, (z[0], z[1]), (z[2], z[3]), (0, 200, 200), -1)
        for d in center_centers:
            cv2.circle(traj_img, d, 6, (0, 0, 255), -1)

        # draw smoothed trajectory (blue)
        pts_full = np.array(list(zip(df["X_px_smooth"].astype(int), df["Y_px_smooth"].astype(int))))
        if pts_full.shape[0] > 1:
            cv2.polylines(traj_img, [pts_full], False, (255, 0, 0), 3)

        cv2.imwrite(traj_png, traj_img)

        # Speed over time (smooth)
        plt.figure(figsize=(10, 4))
        plt.plot(df["Time_s"], df["Speed_px_s_smooth"], label="Speed (px/s)")
        if self.px_per_cm:
            plt.plot(df["Time_s"], df["Speed_cm_s"], label="Speed (cm/s)")
            plt.ylabel("Speed")
        else:
            plt.ylabel("Speed (px/s)")
        plt.xlabel("Time (s)")
        plt.title(f"Speed over Time - {mouse_id}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(speed_png, dpi=300)
        plt.close()

        logging.info("Saved trajectory image to %s", traj_png)
        logging.info("Saved heatmap to %s", heatmap_png)
        logging.info("Saved speed plot to %s", speed_png)
        logging.info("All outputs saved in %s", OUTPUT_DIR)

        return summary

if __name__ == "__main__":
    # Launch GUI entry window
    StartupWindow().mainloop()
