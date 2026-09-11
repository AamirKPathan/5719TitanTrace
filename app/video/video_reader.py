import cv2
from pathlib import Path

class VideoReader:
    def __init__(self,video_path):
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {self.video_path}"
            )
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open video: {self.video_path}"
            )
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        self.duration = (
            self.frame_count / self.fps
            if self.fps > 0
            else 0
        )

    def get_info(self):
        return {
            "filename": self.video_path.name,
            "fps": self.fps,
            "frames": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration,
        }

    def get_frame(self, frame_number):
        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES, frame_number
        )

        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    def get_frame_at_time(self, seconds):
        frame_number = int(seconds * self.fps)
        return self.get_frame(frame_number)

    def release(self):
        self.cap.release()