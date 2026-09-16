import cv2
from pathlib import Path


VIDEO_PATH = Path(
    "data/videos/test.webm"
)

OUTPUT_DIR = Path(
    "data/dataset/raw_frames"
)

INTERVAL_SECONDS = 1.0


def main():

    print("Opening video...")

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    saved = 0
    current_time = 0.0

    while current_time < duration:

        frame_number = int(
            current_time * fps
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        success, frame = cap.read()

        if success:

            output_path = (
                OUTPUT_DIR /
                f"frame_{saved:05d}.jpg"
            )

            cv2.imwrite(
                str(output_path),
                frame
            )

            saved += 1

        current_time += INTERVAL_SECONDS

    cap.release()

    print()
    print("=== Training Frame Extraction ===")
    print(f"Saved: {saved}")
    print(f"Location: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()