from transformers import pipeline
from PIL import Image

import os

class ViTClassifier:
    def __init__(self, model_name="prithivMLmods/Deep-Fake-Detector-v2-Model"):
        # We specify a local project directory so it's portable.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(base_dir, "models", model_name.replace("/", "--"))
        
        # If the model exists locally, it just loads it instantly. 
        # If not, it downloads it and saves it locally.
        if not os.path.exists(self.model_dir):
            print(f"Downloading {model_name} to local project folder...")
            self.pipe = pipeline("image-classification", model=model_name)
            self.pipe.save_pretrained(self.model_dir)
            print(f"Saved weights to {self.model_dir}")
        else:
            print(f"Loading local weights from {self.model_dir}")
            self.pipe = pipeline("image-classification", model=self.model_dir)
    
    def predict(self, images: list[Image.Image]) -> list[dict]:
        """Runs batch prediction. Returns list of highest confidence labels."""
        if not images: return []
        results = self.pipe(images) # Returns a list of lists of dicts
        return [{"label": res[0]['label'], "score": res[0]['score']} for res in results]
