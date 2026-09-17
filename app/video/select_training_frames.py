import cv2
from pathlib import Path


SOURCE_DIR = Path("data/dataset/raw_frames")
OUTPUT_DIR = Path("data/dataset/images/train")

WINDOW_NAME = "TitanTrace Training Frame Selector"


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    frames = sorted(
        SOURCE_DIR.glob("*.jpg")
    )

    if not frames:
        print("No frames found.")
        print(f"Expected frames in: {SOURCE_DIR}")
        return

    print(f"Found {len(frames)} frames.")
    print()
    print("Controls:")
    print("  Y = keep frame")
    print("  N = skip frame")
    print("  Q = quit")
    print()

    selected = 0

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    for index, frame_path in enumerate(frames):

        image = cv2.imread(
            str(frame_path)
        )

        if image is None:
            continue

        display = image.copy()

        # Display frame information.
        text = (
            f"{frame_path.name} | "
            f"{index + 1}/{len(frames)}"
        )

        cv2.rectangle(
            display,
            (0, 0),
            (700, 50),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            display,
            text,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            WINDOW_NAME,
            display
        )

        while True:

            key = cv2.waitKey(0) & 0xFF

            # Y = keep
            if key == ord("y"):
                output_path = (
                    OUTPUT_DIR /
                    frame_path.name
                )

                cv2.imwrite(
                    str(output_path),
                    image
                )

                selected += 1

                print(
                    f"KEEP  {frame_path.name}"
                )

                break

            # N = skip
            elif key == ord("n"):

                print(
                    f"SKIP  {frame_path.name}"
                )

                break

            # Q = quit
            elif key == ord("q"):

                cv2.destroyAllWindows()

                print()
                print(
                    f"Selected {selected} frames."
                )
                print(
                    f"Saved to: {OUTPUT_DIR}"
                )

                return

    cv2.destroyAllWindows()

    print()
    print("=== Selection Complete ===")
    print(f"Selected: {selected}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()