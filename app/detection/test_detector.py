from ultralytics import YOLO
from pathlib import Path
import cv2

MODEL_NAME = "yolo11n.pt"
FRAME_DIR = Path("data/frames")
OUTPUT_DIR = Path("data/detection_results")

def main():
    print("Loading YOLO...")

    model = YOLO(MODEL_NAME)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = sorted(FRAME_DIR.glob("*.jpg"))

    if not frames:
        print("No frames found.")
        return

    print(f"Found {len(frames)} frames.")

    for frame_path in frames:
        print(f"Processing {frame_path.name}")

        image = cv2.imread(str(frame_path))

        if image is None:
            print(f"Could not read {frame_path}")
            continue

        results = model(image)

        annotated  = results[0].plot()

        output_path = OUTPUT_DIR / frame_path.name

        cv2.imwrite(str(output_path), annotated)

    print()
    print("Detection Complete.")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()