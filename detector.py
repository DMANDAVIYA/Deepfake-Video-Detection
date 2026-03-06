from video_processor import extract_frames
from vit_classifier import ViTClassifier

class DeepfakeDetector:
    def __init__(self, classifier: ViTClassifier = None):
        self.classifier = classifier or ViTClassifier()

    def detect(self, video_path: str, num_frames: int = 10, threshold: float = 0.5) -> dict:
        """Analyzes a video and returns fake probability."""
        frames = extract_frames(video_path, num_frames)
        if not frames: return {"is_fake": False, "confidence": 0.0, "details": "No frames extracted"}

        predictions = self.classifier.predict(frames)
        
        fake_scores = [p['score'] for p in predictions if p['label'].lower() == 'deepfake']
        avg_fake_score = sum(fake_scores) / len(predictions) if predictions else 0.0

        return {
            "is_fake": avg_fake_score >= threshold,
            "confidence": avg_fake_score,
            "details": f"Analyzed {len(frames)} frames. Average deepfake score: {avg_fake_score:.2f}"
        }
