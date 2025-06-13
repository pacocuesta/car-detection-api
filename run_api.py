"""
Script to run the FastAPI application
"""
import uvicorn
from src.api import app

if __name__ == "__main__":
    print("Starting Car Detection API...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )