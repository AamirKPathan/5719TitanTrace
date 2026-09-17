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
    print("  B = go back")
    print("  Q = quit")
    print()

    index = 0

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    while index < len(frames):

        frame_path = frames[index]

        image = cv2.imread(
            str(frame_path)
        )

        if image is None:
            index += 1
            continue

        display = image.copy()

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

            # Keep
            if key == ord("y"):

                output_path = (
                    OUTPUT_DIR /
                    frame_path.name
                )

                cv2.imwrite(
                    str(output_path),
                    image
                )

                print(
                    f"KEEP  {frame_path.name}"
                )

                index += 1
                break

            # Skip
            elif key == ord("n"):

                print(
                    f"SKIP  {frame_path.name}"
                )

                index += 1
                break

            # Go back
            elif key == ord("b"):

                if index > 0:
                    index -= 1
                    print(
                        f"BACK  → {frames[index].name}"
                    )
                else:
                    print("Already at first frame.")

                break

            # Quit
            elif key == ord("q"):

                cv2.destroyAllWindows()

                print()
                print("Selection stopped.")
                print(
                    f"Frames currently saved in: "
                    f"{OUTPUT_DIR}"
                )

                return

    cv2.destroyAllWindows()

    saved_count = len(
        list(OUTPUT_DIR.glob("*.jpg"))
    )

    print()
    print("=== Selection Complete ===")
    print(f"Training frames saved: {saved_count}")
    print(f"Location: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()