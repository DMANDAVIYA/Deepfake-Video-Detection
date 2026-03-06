import os
import json
import glob
from detector import DeepfakeDetector

INPUT_DIR = os.path.join("data", "input")
OUTPUT_DIR = os.path.join("data", "output")

def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    video_files = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
    
    if not video_files:
        print(f"No .mp4 files found in {INPUT_DIR}")
        return

    print(f"Initializing detector. Found {len(video_files)} videos to process...")
    detector = DeepfakeDetector() 
    
    results = {}
    
    for video_path in video_files:
        filename = os.path.basename(video_path)
        print(f"Processing {filename}...")
        try:
            res = detector.detect(video_path)
            results[filename] = res
            print(f"  -> {'FAKE' if res['is_fake'] else 'REAL'} (Confidence: {res['confidence']:.2f})")
        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")
            results[filename] = {"error": str(e)}

    output_file = os.path.join(OUTPUT_DIR, "results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nProcessing complete! Results saved to {output_file}")

if __name__ == "__main__":
    main()
