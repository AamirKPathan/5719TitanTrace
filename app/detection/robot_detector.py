from ultralytics import YOLO
from pathlib import Path
import cv2

MODEL_PATH = "models/robot_gamepiece.pt"

CONFIDENCE_THRESHOLD = 0.35

class RobotDetector:
    def __init__(self, model_path=MODEL_PATH):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"CUstom model not found: {model_path}"
            )
        print(f"Loading robot detector: {model_path}")

        self.model = YOLO(str(model_path))

    def detect(self, image):
        """
        Detect robots in a single image.

        Returns a list of dictionaries containing bounding box and confidence information.
        """

        results = self.model(
            image,
            verbose=False
        )

        detections = []
        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(
                float,
                box.xyxy[0]
            )

            # Class 0 = robot
            if class_id == 0:
                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": confidence
                })

        return detections
    def detect_with_image(self, image):
        """
        Detect robots and return an annotated image.
        """

        results = self.model(
            image,
            verbose=False
        )

        annotated = results[0].plot()

        return annotated