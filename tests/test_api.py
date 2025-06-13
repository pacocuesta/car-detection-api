"""
Tests for the FastAPI application
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import tempfile
import os
from PIL import Image
import io

# Import the app
from src.api import app

class TestCarDetectionAPI:
    """Test cases for the Car Detection API"""
    
    @pytest.fixture
    def client(self):
        """Create a test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_image_file(self):
        """Create a sample image file for testing"""
        # Create a simple RGB image
        image = Image.new('RGB', (100, 100), color='red')
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        return img_byte_arr
    
    def test_root_endpoint(self, client):
        """Test the root health check endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Car Detection API is running"
        assert data["status"] == "healthy"
    
    @patch('src.api.detector')
    def test_model_info_endpoint(self, mock_detector, client):
        """Test the model info endpoint"""
        # Mock detector
        mock_detector.get_model_info.return_value = {
            "model_name": "facebook/detr-resnet-50",
            "confidence_threshold": 0.9,
            "available_labels": ["person", "car", "bicycle"]
        }
        
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "facebook/detr-resnet-50"
        assert "car" in data["available_labels"]
    
    @patch('src.api.detector')
    def test_detect_car_endpoint_success(self, mock_detector, client, sample_image_file):
        """Test successful car detection"""
        # Mock detector response
        mock_detector.detect_cars.return_value = (True, [
            {
                "label": "car",
                "confidence": 0.95,
                "bbox": [100, 100, 200, 200],
                "bbox_format": "xyxy"
            }
        ])
        mock_detector.model_name = "facebook/detr-resnet-50"
        mock_detector.confidence_threshold = 0.9
        
        # Make request
        response = client.post(
            "/detect-car",
            files={"file": ("test.jpg", sample_image_file, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["has_cars"] is True
        assert data["car_count"] == 1
        assert len(data["detections"]) == 1
        assert data["detections"][0]["label"] == "car"
    
    @patch('src.api.detector')
    def test_detect_car_endpoint_no_cars(self, mock_detector, client, sample_image_file):
        """Test car detection when no cars are present"""
        # Mock detector response
        mock_detector.detect_cars.return_value = (False, [])
        mock_detector.model_name = "facebook/detr-resnet-50"
        mock_detector.confidence_threshold = 0.9
        
        # Make request
        response = client.post(
            "/detect-car",
            files={"file": ("test.jpg", sample_image_file, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["has_cars"] is False
        assert data["car_count"] == 0
        assert len(data["detections"]) == 0
    
    @patch('src.api.detector')
    def test_detect_car_simple_endpoint(self, mock_detector, client, sample_image_file):
        """Test simple car detection endpoint"""
        # Mock detector response
        mock_detector.detect_car_simple.return_value = True
        
        # Make request
        response = client.post(
            "/detect-car-simple",
            files={"file": ("test.jpg", sample_image_file, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["car_detected"] is True
        assert data["filename"] == "test.jpg"
    
    def test_invalid_file_type(self, client):
        """Test uploading non-image file"""
        # Create a text file
        text_content = b"This is not an image"
        
        response = client.post(
            "/detect-car",
            files={"file": ("test.txt", io.BytesIO(text_content), "text/plain")}
        )
        
        assert response.status_code == 400
        assert "File must be an image" in response.json()["detail"]
    
    @patch('src.api.detector', None)
    def test_model_not_loaded(self, client, sample_image_file):
        """Test API behavior when model is not loaded"""
        response = client.post(
            "/detect-car",
            files={"file": ("test.jpg", sample_image_file, "image/jpeg")}
        )
        
        assert response.status_code == 500
        assert "Model not loaded" in response.json()["detail"]
    
    @patch('src.api.detector')
    def test_detection_error_handling(self, mock_detector, client, sample_image_file):
        """Test error handling during detection"""
        # Mock detector to raise an exception
        mock_detector.detect_cars.side_effect = Exception("Processing error")
        
        response = client.post(
            "/detect-car",
            files={"file": ("test.jpg", sample_image_file, "image/jpeg")}
        )
        
        assert response.status_code == 500
        assert "Error processing image" in response.json()["detail"]