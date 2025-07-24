from PIL import Image, ImageEnhance, ImageFilter
from src.utlis import BASE_DIR
import os

class ImagePreprocessor:
    def __init__(self, brightness=1.2, contrast=1.5, sharpness=1.0, denoise=True, to_grayscale=True):
        self.brightness = brightness
        self.contrast = contrast
        self.sharpness = sharpness
        self.denoise = denoise
        self.to_grayscale = to_grayscale
        self.BASE_DIR = BASE_DIR

    def preprocess(self, image_path):
        image = Image.open(image_path)

        image = ImageEnhance.Brightness(image).enhance(self.brightness)
        image = ImageEnhance.Contrast(image).enhance(self.contrast)
        image = ImageEnhance.Sharpness(image).enhance(self.sharpness)

        if self.denoise:
            image = image.filter(ImageFilter.GaussianBlur(radius=1))

        if self.to_grayscale:
            image = image.convert('L')

        return image