import cv2
from pathlib import Path

class FrameExtractor:
    def __init__(self, video_reader):
        self.video = video_reader
    def extract_every_n_seconds(self, output_dir, interval_seconds=5):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        time = 0.0

        while time < self.video.duration:
            frame = self.video.get_frame_at_time(time)

            if frame is not None:
                filename = output_dir / f"frame_{saved:04d}.jpg"
                cv2.imwrite(str(filename), frame)
                saved+= 1

            time += interval_seconds
        return saved