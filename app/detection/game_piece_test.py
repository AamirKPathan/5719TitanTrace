from ultralytics import YOLO
from pathlib import Path
import cv2

MODEL_NAME = "yolo11n.pt"
FRAME_DIR = Path("data/frames")
OUTPUT_DIR = Path("data/game_piece_results")

SPORTS_BALL_CLASS = 32
CONFIDENCE_THRESHOLD = 0.20

def main():
    print("Loading YOLO...")

    model = YOLO(MODEL_NAME)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = sorted(FRAME_DIR.glob("*.jpg"))

    if not frames:
        print("No frames found.")
        return

    detected_count = 0

    for frame_path in frames:
        image = cv2.imread(str(frame_path))

        if image is None:
            continue

        results = model(image, verbose = False)

        result = results[0]
        annotated = image.copy()

        found = False

        if result.boxes is not None:
            for box in result.boxes: class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if (
                class_id == SPORTS_BALL_CLASS
                and confidence >= CONFIDENCE_THRESHOLD
            ):
                found = True

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )
                label = f"game-piece? {confidence:.2f}"

                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            if found:
                detected_count += 1
                output_path = OUTPUT_DIR / frame_path.name
                cv2.imwrite(
                    str(output_path),
                    annotated
                )
        print()
        print("=== Game Piece Experiment ===")
        print(f"Frames analyzed: {len(frames)}")
        print(f"Frames with detections: {detected_count}")
        print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()