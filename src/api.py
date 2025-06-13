"""
FastAPI application for car detection
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from .car_detector import CarDetector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Car Detection API",
    description="API to detect cars in images using DETR model",
    version="1.0.0"
)

# Initialize car detector
try:
    detector = CarDetector()
    logger.info("Car detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize car detector: {e}")
    detector = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Car Detection API is running", "status": "healthy"}

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if not detector:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return detector.get_model_info()

@app.post("/detect-car")
async def detect_car(file: UploadFile = File(...)):
    """
    Detect cars in uploaded image
    
    Args:
        file: Image file to analyze
        
    Returns:
        JSON response with detection results
    """
    if not detector:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Detect cars
        has_cars, detections = detector.detect_cars(tmp_file_path)
        
        return JSONResponse(content={
            "filename": file.filename,
            "has_cars": has_cars,
            "car_count": len(detections),
            "detections": detections,
            "model_info": {
                "model_name": detector.model_name,
                "confidence_threshold": detector.confidence_threshold
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.post("/detect-car-simple")
async def detect_car_simple(file: UploadFile = File(...)):
    """
    Simple car detection endpoint that returns only boolean result
    
    Args:
        file: Image file to analyze
        
    Returns:
        JSON response with simple boolean result
    """
    if not detector:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Detect cars
        has_cars = detector.detect_car_simple(tmp_file_path)
        
        return JSONResponse(content={
            "filename": file.filename,
            "car_detected": has_cars
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)