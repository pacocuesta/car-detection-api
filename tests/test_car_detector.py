"""
Tests for the CarDetector class
"""
import pytest
import tempfile
import os
from PIL import Image
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.car_detector import CarDetector

class TestCarDetector:
    """Test cases for CarDetector"""
    
    @pytest.fixture
    def mock_detector(self):
        """Create a mock detector for testing"""
        with patch('src.car_detector.DetrImageProcessor') as mock_processor, \
             patch('src.car_detector.DetrForObjectDetection') as mock_model:
            
            # Mock processor
            mock_processor_instance = Mock()
            mock_processor.from_pretrained.return_value = mock_processor_instance
            
            # Mock model
            mock_model_instance = Mock()
            mock_model_instance.config.id2label = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle'}
            mock_model.from_pretrained.return_value = mock_model_instance
            
            detector = CarDetector()
            detector.processor = mock_processor_instance
            detector.model = mock_model_instance
            
            return detector, mock_processor_instance, mock_model_instance
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        # Create a simple RGB image
        image = Image.new('RGB', (640, 480), color='blue')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image.save(tmp_file.name)
            yield tmp_file.name
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    def test_detector_initialization(self, mock_detector):
        """Test that detector initializes correctly"""
        detector, mock_processor, mock_model = mock_detector
        
        assert detector.model_name == "facebook/detr-resnet-50"
        assert detector.confidence_threshold == 0.9
        assert detector.processor is not None
        assert detector.model is not None
    
    def test_detect_cars_with_car_present(self, mock_detector, sample_image):
        """Test car detection when car is present"""
        detector, mock_processor, mock_model = mock_detector
        
        # Mock the model outputs
        mock_outputs = Mock()
        mock_model.return_value = mock_outputs
        
        # Mock post-processing results
        mock_results = {
            "scores": [0.95],  # High confidence
            "labels": [2],     # Car label
            "boxes": [[100, 100, 200, 200]]  # Bounding box
        }
        mock_processor.post_process_object_detection.return_value = [mock_results]
        
        # Mock processor input processing
        mock_processor.return_value = {"pixel_values": Mock()}
        
        # Test detection
        has_cars, detections = detector.detect_cars(sample_image)
        
        assert has_cars is True
        assert len(detections) == 1
        assert detections[0]["label"] == "car"
        assert detections[0]["confidence"] == 0.95
        assert detections[0]["bbox"] == [100, 100, 200, 200]
    
    def test_detect_cars_no_car_present(self, mock_detector, sample_image):
        """Test car detection when no car is present"""
        detector, mock_processor, mock_model = mock_detector
        
        # Mock the model outputs
        mock_outputs = Mock()
        mock_model.return_value = mock_outputs
        
        # Mock post-processing results with no cars
        mock_results = {
            "scores": [0.95],  # High confidence
            "labels": [0],     # Person label (not car)
            "boxes": [[100, 100, 200, 200]]
        }
        mock_processor.post_process_object_detection.return_value = [mock_results]
        
        # Mock processor input processing
        mock_processor.return_value = {"pixel_values": Mock()}
        
        # Test detection
        has_cars, detections = detector.detect_cars(sample_image)
        
        assert has_cars is False
        assert len(detections) == 0
    
    def test_detect_car_simple(self, mock_detector, sample_image):
        """Test simple car detection method"""
        detector, mock_processor, mock_model = mock_detector
        
        # Mock the detect_cars method
        with patch.object(detector, 'detect_cars') as mock_detect:
            mock_detect.return_value = (True, [{"label": "car"}])
            
            result = detector.detect_car_simple(sample_image)
            assert result is True
            
            mock_detect.return_value = (False, [])
            result = detector.detect_car_simple(sample_image)
            assert result is False
    
    def test_file_not_found_error(self, mock_detector):
        """Test handling of non-existent file"""
        detector, _, _ = mock_detector
        
        with pytest.raises(FileNotFoundError):
            detector.detect_cars("non_existent_file.jpg")
    
    def test_get_model_info(self, mock_detector):
        """Test model info retrieval"""
        detector, _, _ = mock_detector
        
        info = detector.get_model_info()
        
        assert info["model_name"] == "facebook/detr-resnet-50"
        assert info["confidence_threshold"] == 0.9
        assert "available_labels" in info
        assert "car" in info["available_labels"]
    
    def test_image_format_conversion(self, mock_detector):
        """Test that non-RGB images are converted properly"""
        detector, mock_processor, mock_model = mock_detector
        
        # Create a grayscale image
        gray_image = Image.new('L', (100, 100), color=128)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            gray_image.save(tmp_file.name)
            
            # Mock the model outputs
            mock_outputs = Mock()
            mock_model.return_value = mock_outputs
            
            # Mock post-processing results
            mock_results = {"scores": [], "labels": [], "boxes": []}
            mock_processor.post_process_object_detection.return_value = [mock_results]
            mock_processor.return_value = {"pixel_values": Mock()}
            
            # This should not raise an error
            has_cars, detections = detector.detect_cars(tmp_file.name)
            
            # Cleanup
            os.unlink(tmp_file.name)
    
    def test_multiple_cars_detection(self, mock_detector, sample_image):
        """Test detection of multiple cars"""
        detector, mock_processor, mock_model = mock_detector
        
        # Mock the model outputs
        mock_outputs = Mock()
        mock_model.return_value = mock_outputs
        
        # Mock post-processing results with multiple cars
        mock_results = {
            "scores": [0.95, 0.87, 0.92],
            "labels": [2, 2, 0],  # Two cars and one person
            "boxes": [[100, 100, 200, 200], [300, 300, 400, 400], [500, 500, 600, 600]]
        }
        mock_processor.post_process_object_detection.return_value = [mock_results]
        mock_processor.return_value = {"pixel_values": Mock()}
        
        # Test detection
        has_cars, detections = detector.detect_cars(sample_image)
        
        assert has_cars is True
        assert len(detections) == 2  # Only cars, not person
        assert all(d["label"] == "car" for d in detections)