import cv2
from PIL import Image

def extract_frames(video_path: str, num_frames: int = 10) -> list[Image.Image]:
    """Extract evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: raise ValueError("Video has zero frames")
    
    frames = []
    step = max(1, total_frames // num_frames)
    
    for i in range(0, total_frames, step):
        if len(frames) >= num_frames: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret: frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        
    cap.release()
    return frames
