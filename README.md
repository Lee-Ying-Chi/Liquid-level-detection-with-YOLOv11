# Surface Height Client

This project uses an Intel RealSense camera and a YOLO segmentation model to estimate liquid surface height inside a bottle.  
The current pipeline detects the bottle and liquid region, extracts a 1D profile from the bottle ROI, fits the liquid surface with RANSAC, and converts the bottle-bottom-to-surface pixel gap into an estimated `ml` value.

## Current Status

- Main runtime entry: `main.py`
- Simple camera preview only: `opencam.py`
- YOLO model path is configured in `config.py`
- ArUco-related code is currently kept in the repo but disabled in the main runtime

## Main Features

- RealSense live camera input
- YOLO segmentation for `bottle` and `Liquid`
- ROI-based image processing for liquid surface estimation
- RANSAC-based liquid surface fitting
- Pixel-gap-to-ml conversion using a configurable lookup table
- Terminal telemetry output
- WebSocket client for sending averaged `ml` values

## Project Structure

- [main.py](/c:/Users/Emma/Documents/Surface_Height_Client/main.py): Main application loop
- [config.py](/c:/Users/Emma/Documents/Surface_Height_Client/config.py): Runtime parameters and calibration values
- [yolo_roi.py](/c:/Users/Emma/Documents/Surface_Height_Client/yolo_roi.py): YOLO inference result parsing
- [image_proc.py](/c:/Users/Emma/Documents/Surface_Height_Client/image_proc.py): Sobel/profile/RANSAC helper functions
- [telemetry.py](/c:/Users/Emma/Documents/Surface_Height_Client/telemetry.py): Terminal telemetry and CSV logging
- [Surface_Height_Client.py](/c:/Users/Emma/Documents/Surface_Height_Client/Surface_Height_Client.py): WebSocket client
- [opencam.py](/c:/Users/Emma/Documents/Surface_Height_Client/opencam.py): Minimal camera preview

## Requirements

The project dependencies are listed in [requirements.txt](/c:/Users/Emma/Documents/Surface_Height_Client/requirements.txt).

Core runtime dependencies:

- Python 3.10 or newer recommended
- `numpy`
- `opencv-contrib-python`
- `pyrealsense2`
- `torch`
- `torchvision`
- `ultralytics`
- `open3d`
- `websockets`

## Installation

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Model Setup

The YOLO weight path is configured in [config.py](/c:/Users/Emma/Documents/Surface_Height_Client/config.py):

```python
YOLO_WEIGHTS = "./Yolo/greenBottle0319/best.pt"
```

Make sure the model file exists at that path before running the main program.

## Running

Run the main application:

```powershell
python main.py
```

Run the camera preview only:

```powershell
python opencam.py
```

## Runtime Controls

`main.py` supports these keys:

- `1`: RGB view
- `2`: 3D point cloud view
- `3`: Line / segmentation view
- `4`: Profile / RANSAC view
- `q` or `Esc`: Quit

## Runtime Output

The application provides:

- OpenCV windows for live visualization
- Terminal telemetry, including:
  - `y_vertex`
  - `bottom_gap_px`
  - `ml`
  - `avg_ml`
  - `inliers`
- CSV logging under the `logs/` directory

## Configuration Notes

Most runtime behavior is controlled from [config.py](/c:/Users/Emma/Documents/Surface_Height_Client/config.py).

Important parameters:

- `YOLO_WEIGHTS`
- `YOLO_CONF`
- `PEAK_TH`
- `RANSAC_Y_WIN`
- `RANSAC_EDGE_TH`
- `RANSAC_INLIER_TH`
- `RANSAC_MIN_INLIERS`
- `BOTTOM_GAP_PX_TO_ML`

`BOTTOM_GAP_PX_TO_ML` is the current temporary mapping from pixel gap to milliliters.  
If you change bottle geometry, camera position, or scene scale, this table should be recalibrated.

## Notes

- This project currently assumes the YOLO model contains at least the labels `bottle` and `Liquid`.
- ArUco code is still present in the repository for future use, but it is not part of the current main pipeline.
- If RealSense cannot be opened, confirm camera connection and the installed `pyrealsense2` version.

