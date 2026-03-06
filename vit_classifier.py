from transformers import pipeline
from PIL import Image

class ViTClassifier:
    def __init__(self, model_name="prithivMLmods/Deep-Fake-Detector-v2-Model"):
        # The pipeline automatically downloads and caches the model weights and config 
        # from the HuggingFace Hub the first time it is instantiated.
        self.pipe = pipeline("image-classification", model=model_name)
    
    def predict(self, images: list[Image.Image]) -> list[dict]:
        """Runs batch prediction. Returns list of highest confidence labels."""
        if not images: return []
        results = self.pipe(images) # Returns a list of lists of dicts
        return [{"label": res[0]['label'], "score": res[0]['score']} for res in results]
