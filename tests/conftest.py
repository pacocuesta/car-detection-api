"""
Pytest configuration and shared fixtures
"""
import pytest
import tempfile
import os
from PIL import Image

@pytest.fixture(scope="session")
def test_images_dir():
    """Create a temporary directory for test images"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup is handled by the OS for temp directories

@pytest.fixture
def car_image(test_images_dir):
    """Create a test image that should contain a car (mock)"""
    image = Image.new('RGB', (640, 480), color='blue')
    image_path = os.path.join(test_images_dir, 'car_test.jpg')
    image.save(image_path)
    return image_path

@pytest.fixture
def no_car_image(test_images_dir):
    """Create a test image that should not contain a car"""
    image = Image.new('RGB', (640, 480), color='green')
    image_path = os.path.join(test_images_dir, 'no_car_test.jpg')
    image.save(image_path)
    return image_path