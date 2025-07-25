### Importing local modules
from src.utlis import BASE_DIR
from src.preprocessing.label_creator import label_creator
from src.preprocessing.image_processing import ImagePreprocessor
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.generate_dataset import generate_dataset
from src.preprocessing.raw_data_processing import DataProcessing
from src.model.save_load import save
from src.model.save_load import load
from src.model.train import design_train_save_model

### Preprocessing
# create label dataset for images
label_creator()
# image processing, feature extraction and create raw dataset
generate_dataset()
# data processing and split train test data
data_processing = DataProcessing(n_components=100)
data_processing.transform()
