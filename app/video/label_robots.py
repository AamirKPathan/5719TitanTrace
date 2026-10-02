import cv2
from pathlib import Path

IMAGE_DIR = Path("data/dataset/images/train")
LABEL_DIR = Path("data/dataset/labels/train")

CLASS_ID = 0

drawing = False
start_x = 0
start_y = 0
current_box = None
boxes = []

def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, current_box

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x = x
        start_y = y
        current_box = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_box = (
            start_x,
            start_y,
            x,
            y
        )

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        x1 = min(start_x, x)
        y1 = min(start_y, y)
        x2 = max(start_x, x)
        y2 = max(start_y, y)

        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            boxes.append((x1, y1, x2, y2))

        current_box = None

def save_labels(image_path, image_width, image_height):
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    with open(label_path, "w") as file:
        for x1, y1, x2, y2 in boxes:

            center_x = ((x1 + x2) / 2) / image_width
            center_y = ((y1 + y2) / 2) / image_height

            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height

            file.write(
                f"{CLASS_ID} "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
            )

def main():
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(IMAGE_DIR.glob("*.jpg")) +
        list(IMAGE_DIR.glob("*.png")) +
        list(IMAGE_DIR.glob("*.jpeg"))
    )

    if not images:
        print("No training images found.")
        return

    print(f"Found {len(images)} images.")

    index = 0

    cv2.namedWindow("TitanTrace Robot Labeler")
    cv2.setMouseCallback(
        "TitanTrace Robot Labeler",
        mouse_callback
    )

    while 0 <= index < len(images):

        global boxes
        boxes = []

        image_path = images[index]
        image = cv2.imread(str(image_path))

        if image is None:
            index += 1
            continue

        height, width = image.shape[:2]

        print()
        print(f"Image {index + 1}/{len(images)}: {image_path.name}")

        while True:

            display = image.copy()

            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

            if current_box is not None:
                x1, y1, x2, y2 = current_box

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

            cv2.putText(
                display,
                f"{index + 1}/{len(images)}  Boxes: {len(boxes)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "TitanTrace Robot Labeler",
                display
            )

            key = cv2.waitKey(20) & 0xFF

            if key == ord("n"):
                save_labels(
                    image_path,
                    width,
                    height
                )
                index += 1
                break

            elif key == ord("b"):
                index = max(0, index - 1)
                break

            elif key == ord("u"):
                if boxes:
                    boxes.pop()

            elif key == ord("q"):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

    print()
    print("Labeling complete.")

if __name__ == "__main__":
    main()