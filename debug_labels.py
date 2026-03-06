from vit_classifier import ViTClassifier
from video_processor import extract_frames

frames = extract_frames("data/input/YTDown.com_YouTube_How-To-Walk-Like-a-Male-Model-in-Under-a_Media_g8fSLV0sIbQ_002_720p.mp4")
clf = ViTClassifier()

import sys
# Print raw results for first 5 frames
raw = clf.pipe(frames[:5])
for i, r in enumerate(raw):
    print(f"Frame {i}: {r}")
