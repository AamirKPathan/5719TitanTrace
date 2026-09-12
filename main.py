from app.video.video_reader import VideoReader

VIDEO_PATH = "data/videos/test.webm"
FRAME_OUTPUT_DIR = "data/frames"

def main():
    print("Starting TitanTrace...")
    video = VideoReader(VIDEO_PATH)
    info = video.get_info()

    print("\n=== TitanTrace Video Information ===")
    print(f"File: {info['filename']}")
    print(f"FPS: {info['fps']:.2f}")
    print(f"Frames: {info['frames']}")
    print(f"Width: {info['width']}x{info['height']}")
    print(f"Duration: {info['duration_seconds']:.2f} seconds")

    print("\nExtracting frames...")
    extractor = FrameExtractor(video)

    saved = extractor.extract_every_n_seconds(
        FRAME_OUTPUT_DIR,
        interval_seconds=5
    )

    print(f"\nSaved {saved} frames to {FRAME_OUTPUT_DIR}.")

    video.release()

if __name__ == "__main__":
    main()