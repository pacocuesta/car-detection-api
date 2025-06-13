"""
Car Detection Module using DETR (Detection Transformer)
"""
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import logging
from typing import Tuple, List, Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CarDetector:
    """Car detection using Facebook's DETR model"""
    
    def __init__(self, model_name: str = "facebook/detr-resnet-50", confidence_threshold: float = 0.9):
        """
        Initialize the car detector
        
        Args:
            model_name: HuggingFace model identifier
            confidence_threshold: Minimum confidence score for detections
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the DETR model and processor"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            self.model = DetrForObjectDetection.from_pretrained(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_cars(self, image_path: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Detect cars in an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (has_cars: bool, detections: List[Dict])
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load and process image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes, 
                threshold=self.confidence_threshold
            )[0]
            
            # Extract car detections
            car_detections = []
            has_cars = False
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                label_name = self.model.config.id2label[label.item()]
                if label_name == "car":
                    has_cars = True
                    car_detections.append({
                        "label": label_name,
                        "confidence": score.item(),
                        "bbox": box.tolist(),
                        "bbox_format": "xyxy"  # x1, y1, x2, y2
                    })
            
            logger.info(f"Detected {len(car_detections)} cars in {image_path}")
            return has_cars, car_detections
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def detect_car_simple(self, image_path: str) -> bool:
        """
        Simple car detection that returns only boolean result
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if cars are detected, False otherwise
        """
        has_cars, _ = self.detect_cars(image_path)
        return has_cars
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "available_labels": list(self.model.config.id2label.values()) if self.model else None
        }