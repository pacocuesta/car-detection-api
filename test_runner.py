"""
Simple test runner script for manual testing
"""
import sys
import os
from src.car_detector import CarDetector
from PIL import Image
import tempfile

def create_test_image():
    """Create a simple test image"""
    image = Image.new('RGB', (640, 480), color='blue')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        image.save(tmp_file.name)
        return tmp_file.name

def test_car_detector():
    """Test the car detector with a sample image"""
    print("Testing Car Detector...")
    
    try:
        # Initialize detector
        print("Initializing detector...")
        detector = CarDetector()
        
        # Get model info
        info = detector.get_model_info()
        print(f"Model: {info['model_name']}")
        print(f"Confidence threshold: {info['confidence_threshold']}")
        print(f"Available labels: {len(info['available_labels'])} labels")
        
        # Create test image
        test_image_path = create_test_image()
        print(f"Created test image: {test_image_path}")
        
        # Test detection
        print("Running detection...")
        has_cars, detections = detector.detect_cars(test_image_path)
        
        print(f"Cars detected: {has_cars}")
        print(f"Number of detections: {len(detections)}")
        
        for i, detection in enumerate(detections):
            print(f"Detection {i+1}:")
            print(f"  Label: {detection['label']}")
            print(f"  Confidence: {detection['confidence']:.3f}")
            print(f"  Bounding box: {detection['bbox']}")
        
        # Test simple detection
        simple_result = detector.detect_car_simple(test_image_path)
        print(f"Simple detection result: {simple_result}")
        
        # Cleanup
        os.unlink(test_image_path)
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_car_detector()
    sys.exit(0 if success else 1)