from detector import DeepfakeDetector
import os
import cv2
import numpy as np

# Create a blank dummy video for the test to process
dummy_video = "dummy_test.mp4"
out = cv2.VideoWriter(dummy_video, cv2.VideoWriter_fourcc(*'mp4v'), 1.0, (224, 224))
for _ in range(3):
    out.write(np.zeros((224, 224, 3), dtype=np.uint8))
out.release()

print("Initializing detector. This WILL download the model weights (approx 340MB)...")
detector = DeepfakeDetector() # First init triggers the HuggingFace download

print("Running detection on dummy video...")
result = detector.detect(dummy_video)
print(f"Result: {result}")

# Clean up
if os.path.exists(dummy_video):
    os.remove(dummy_video)
