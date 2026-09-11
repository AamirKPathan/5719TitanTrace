from app.video.video_reader import VideoReader

VIDEO_PATH = "data/videos/test.mp4"

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

    video.release()

if __name__ == "__main__":
    main()