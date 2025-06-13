# Car Detection API

API to detect cars in images using Facebook's DETR (Detection Transformer) model.

## Features

- **Car Detection**: Detect cars in uploaded images using state-of-the-art AI
- **REST API**: FastAPI-based web service with automatic documentation
- **High Accuracy**: Uses Facebook's DETR ResNet-50 model
- **Comprehensive Testing**: Full test suite with pytest
- **Easy Deployment**: Ready for production deployment

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run tests with coverage
python -m pytest tests/ -v --cov=src

# Run specific test file
python -m pytest tests/test_car_detector.py -v
```

### 3. Test the Core Functionality

```bash
# Quick test of the car detector
python test_runner.py
```

### 4. Start the API Server

```bash
# Start the FastAPI server
python run_api.py
```

The API will be available at:
- **API Base**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## API Endpoints

### Health Check
```
GET /
```
Returns API status and health information.

### Model Information
```
GET /model-info
```
Returns information about the loaded AI model.

### Car Detection (Detailed)
```
POST /detect-car
```
Upload an image and get detailed car detection results including:
- Whether cars are present
- Number of cars detected
- Confidence scores
- Bounding box coordinates

### Car Detection (Simple)
```
POST /detect-car-simple
```
Upload an image and get a simple boolean result indicating if cars are present.

## Testing Your Code

### 1. Unit Tests
Run the comprehensive test suite:

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run tests with coverage report
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test categories
python -m pytest tests/test_car_detector.py -v  # Core functionality
python -m pytest tests/test_api.py -v          # API endpoints
```

### 2. Manual Testing
Use the test runner for quick manual verification:

```bash
python test_runner.py
```

### 3. API Testing
Start the server and test the endpoints:

```bash
# Start the server
python run_api.py

# In another terminal, test with curl:
curl -X POST "http://localhost:8000/detect-car" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```

### 4. Interactive Testing
Visit http://localhost:8000/docs for interactive API documentation where you can:
- Upload test images directly
- See real-time results
- Explore all available endpoints

## Project Structure

```
car-detection-api/
├── src/
│   ├── __init__.py
│   ├── car_detector.py      # Core detection logic
│   └── api.py              # FastAPI application
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # Test configuration
│   ├── test_car_detector.py # Core functionality tests
│   └── test_api.py         # API endpoint tests
├── sample_images/          # Directory for test images
├── requirements.txt        # Python dependencies
├── test_runner.py         # Manual test script
├── run_api.py            # API server launcher
└── README.md
```

## Model Details

- **Model**: Facebook DETR ResNet-50
- **Task**: Object Detection
- **Classes**: 91 COCO classes (including cars)
- **Confidence Threshold**: 0.9 (configurable)
- **Input**: RGB images of any size
- **Output**: Bounding boxes, labels, and confidence scores

## Testing Strategies

### 1. **Unit Testing**
- Test core detection logic
- Mock external dependencies
- Validate error handling
- Test edge cases

### 2. **Integration Testing**
- Test API endpoints
- Validate request/response formats
- Test file upload handling
- Error response validation

### 3. **Performance Testing**
- Image processing speed
- Memory usage
- Concurrent request handling

### 4. **Manual Testing**
- Real image testing
- Visual result validation
- User experience testing

## Example Usage

### Python Code
```python
from src.car_detector import CarDetector

# Initialize detector
detector = CarDetector()

# Detect cars in an image
has_cars, detections = detector.detect_cars('path/to/image.jpg')

print(f"Cars detected: {has_cars}")
for detection in detections:
    print(f"Car found with {detection['confidence']:.2f} confidence")
```

### API Usage
```python
import requests

# Upload image for detection
with open('car_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect-car',
        files={'file': f}
    )
    
result = response.json()
print(f"Cars detected: {result['has_cars']}")
```

## Development

### Adding New Tests
1. Create test files in the `tests/` directory
2. Use pytest fixtures for common setup
3. Mock external dependencies
4. Test both success and error cases

### Running Continuous Tests
```bash
# Watch for file changes and re-run tests
python -m pytest tests/ -f --tb=short
```

### Code Coverage
```bash
# Generate coverage report
python -m pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html to view detailed coverage
```

This setup provides a robust testing framework for your car detection code with both automated tests and manual testing capabilities.