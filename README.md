# Open Source Rodent Tracking for Open Field Assay Analysis

This project implements a complete Open Field Test (OFT) analysis pipeline using:

- YOLOv8 (Ultralytics) for mouse detection and tracking
- CustomTkinter GUI for the interface
- OpenCV for arena drawing and video processing
- NumPy / Pandas / SciPy for data processing and smoothing
- Matplotlib / Seaborn for heatmaps and graphs
- uv package manager for environment and dependency management

----------------------------------------------------------------
USING `uv` IN THIS PROJECT
----------------------------------------------------------------

The project uses uv, a fast Python package/environment manager.
You do not need pip; uv handles dependency installation.

Prerequisites:
Install uv:
    curl -LsSf https://astral.sh/uv/install.sh | sh
or:
    brew install uv

Verify:
    uv --version

----------------------------------------------------------------
SETTING UP THE PROJECT
----------------------------------------------------------------

1. Create the virtual environment:
    uv venv

2. Install all dependencies:
    uv sync

3. (Optional) Activate environment manually:
    source .venv/bin/activate

----------------------------------------------------------------
RUNNING THE APPLICATION
----------------------------------------------------------------

Start the OFT GUI:
    uv run python main.py

You will see a Startup Window with two choices:
- Live Camera (webcam or USB camera)
- Video File

Select one to continue.

----------------------------------------------------------------
HOW TO USE THE OFT TRACKER
----------------------------------------------------------------

1. Select input source:
   - Camera mode: enter camera ID (0 is default)
   - Video mode: choose a video file (.mp4, .avi, etc.)

2. Enter Mouse ID:
   The Mouse ID appears in results, file naming, and metadata.

3. (Optional) Enable Calibration:
   Check “Calibrate (px → cm)” if you want units converted to centimeters.
   You will be prompted to enter the arena width (in cm).

----------------------------------------------------------------
DRAWING THE ARENA
----------------------------------------------------------------

The first frame of the video is displayed.

- Click multiple points to outline the arena polygon.
- Press ENTER to finish (minimum 3 points).
- Press “r” to reset and redraw.

The software automatically generates:
- 4×4 grid
- Central 2×2 grid region (center zone)

----------------------------------------------------------------
PROCESSING & LIVE TRACKING
----------------------------------------------------------------

Once processing starts:
- YOLOv8 detects the mouse per frame.
- The largest bounding box is selected.
- Mouse center point is extracted.
- Arena, grid, and center zones are drawn.
- Recent trajectory is shown.
- Press “q” to stop early.

Data collected:
- Position (X,Y)
- Speed (raw and smoothed)
- Distance traveled
- Zone classification (center/border)
- Center entries and latency
- Thigmotaxis (time near walls)

----------------------------------------------------------------
OUTPUT FILES (IN OFT_Results/)
----------------------------------------------------------------

For Mouse ID = Mouse001, outputs include:

1. Mouse001_tracked.mp4
   Annotated tracking video with live bounding box and trajectory.

2. Mouse001_PerFrame.csv
   Position, speed, zone, timestamps, frame-by-frame metrics.

3. Mouse001_Summary.xlsx
   Sheet 1: Behavioral summary
   Sheet 2: Per-frame data

4. Mouse001_metadata.json
   Includes model path, fps, resolution, timestamps, summary metrics.

5. Mouse001_Heatmap.png
   Heatmap of mouse occupancy.

6. Mouse001_Trajectory.png
   Full trajectory plotted on arena with center zones.

7. Mouse001_SpeedTimeSeries.png
   Plot of smoothed speed over time.

----------------------------------------------------------------
METRICS INCLUDED
----------------------------------------------------------------

GENERAL
- Total time
- Frames processed
- Path length (px or cm)
- Total distance traveled
- Smoothed vs raw trajectory

SPEED
- Mean speed
- Median speed
- Maximum speed
- Instantaneous pixel/second speed

CENTER-RELATED
- Time spent in center
- Percent time in center
- Number of center entries
- Latency to first entry

THIGMOTAXIS
- Time near walls
- Percentage of total trial spent near perimeter

----------------------------------------------------------------
ADDING DEPENDENCIES
----------------------------------------------------------------

To add a package:
    uv add <package-name>

To add a dev-only package:
    uv add --dev <package-name>

----------------------------------------------------------------
UPDATING DEPENDENCIES
----------------------------------------------------------------

Update one dependency:
    uv up <package>

Update all:
    uv up

----------------------------------------------------------------
TROUBLESHOOTING
----------------------------------------------------------------

YOLO model fails to load:
- Ensure the model file exists at models/YoloV8n_mouse.pt

Camera not opening:
- Try camera ID 0 or 1
- Check OS permissions

No detections:
- Improve lighting
- Lower confidence threshold in code
- Check video resolution

Short videos cause smoothing errors:
- Script automatically falls back to unsmoothed values

----------------------------------------------------------------

