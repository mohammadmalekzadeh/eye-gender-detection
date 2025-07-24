import numpy as np
import pandas as pd
import os
from src.preprocessing.image_processing import ImagePreprocessor
from src.utlis import BASE_DIR

class FeatureExtractor:
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.BASE_DIR = BASE_DIR

    def transform(self, image_path, label=None):
        image = self.preprocessor.preprocess(image_path)

        arr = np.asarray(image).flatten()

        data = {f'pixel_{i}': val for i, val in enumerate(arr)}
        
        if label is not None:
            data['gender'] = label

        return pd.DataFrame([data])