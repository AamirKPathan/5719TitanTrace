# Titan Trace

Titan Trace extracts training frames and prepares robot detections for YOLO training.

## Current Workflow

1. Put source videos in `data/videos`.
2. Extract one frame per second into `data/dataset/raw_frames`:

	```powershell
	.venv\Scripts\python.exe app\video\extract_training_frames.py
	```

3. Put the images you want to label in `data/dataset/images/train`.
4. Start the interactive robot labeler:

	```powershell
	.venv\Scripts\python.exe app\video\label_robots.py
	```

The labeler opens each training image in an OpenCV window. Click and drag around every robot. Press `N` to save the current image and move forward, `B` to go back, `U` to remove the most recently drawn box, and `Q` to quit.

Labels are written automatically to `data/dataset/labels/train` using YOLO format. Every box is class `0` (`robot`) and is stored as:

```text
0 center_x center_y width height
```

Coordinates are normalized to the image dimensions. Images can be `.jpg`, `.jpeg`, or `.png`.

## Environment

Install the Python dependencies with:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
