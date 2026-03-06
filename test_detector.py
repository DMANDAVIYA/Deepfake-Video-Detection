import pytest
from unittest.mock import patch, MagicMock
from detector import DeepfakeDetector
from PIL import Image

@pytest.fixture
def mock_classifier():
    classifier = MagicMock()
    return classifier

def test_detect_all_fake(mock_classifier):
    # Mock extract_frames to return dummy images
    with patch('detector.extract_frames') as mock_extract:
        mock_extract.return_value = [Image.new('RGB', (10, 10))] * 5
        # Mock predict to return "Deepfake" for all frames
        mock_classifier.predict.return_value = [{'label': 'Deepfake', 'score': 0.9}] * 5
        
        detector = DeepfakeDetector(classifier=mock_classifier)
        result = detector.detect("dummy.mp4")
        
        assert result['is_fake'] is True
        assert result['confidence'] == 0.9

def test_detect_mostly_real(mock_classifier):
    with patch('detector.extract_frames') as mock_extract:
        mock_extract.return_value = [Image.new('RGB', (10, 10))] * 5
        # 4 Real, 1 Fake -> average fake score is 0.18 (below threshold)
        predictions = [{'label': 'Realism', 'score': 0.8}] * 4 + [{'label': 'Deepfake', 'score': 0.9}]
        mock_classifier.predict.return_value = predictions
        
        detector = DeepfakeDetector(classifier=mock_classifier)
        result = detector.detect("dummy.mp4")
        
        assert result['is_fake'] is False
        assert result['confidence'] == 0.9 / 5 # 0.18

def test_no_frames_extracted(mock_classifier):
    with patch('detector.extract_frames') as mock_extract:
        mock_extract.return_value = []
        
        detector = DeepfakeDetector(classifier=mock_classifier)
        result = detector.detect("empty.mp4")
        
        assert result['is_fake'] is False
        assert result['confidence'] == 0.0
        assert "No frames" in result['details']
