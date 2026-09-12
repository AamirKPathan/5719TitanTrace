import cv2
from pathlib import Path
from PIL import Image, ImageDraw

VIDEO_PATH = "data/videos/test.webm"
OUTPUT_PATH = "data/contact_sheet.jpg"

video = cv2.VideoCapture(VIDEO_PATH)

fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

times = [
    duration * i /24
    for i in range(24)
]

images = []

for t in times:
    video.set(cv2.CAP_PROP_POS_MSEC, t * 1000)

    success, frame = video.read()
    if not success:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    image.thumbnail((480, 270))

    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 150, 25), fill = "black")
    draw.text((5, 5), f"{t:.1f}s", fill = "white")
    images.append(image)
video.release()

sheet = Image.new("RGB", (960, 2160), "black")

for i, image in enumerate(images):
    x = (i % 2) * 480
    y = (i // 2) * 180

    sheet.paste(image, (x, y))

sheet.save(OUTPUT_PATH, quality=85)

print(f"Created: {OUTPUT_PATH}")