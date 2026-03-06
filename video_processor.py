import cv2
from PIL import Image

def extract_frames(video_path: str) -> list[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: raise ValueError("Video has zero frames")

    frame_indices = [int(i * fps) for i in range(int(total_frames / fps))]

    frames = []
    for idx in frame_indices:
        if idx >= total_frames: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret: frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()
    return frames
