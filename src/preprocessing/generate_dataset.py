import pandas as pd
from tqdm import tqdm
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.image_processing import ImagePreprocessor
from src.utlis import BASE_DIR

def generate_dataset(output_raw=BASE_DIR+'/data/raw/processed_data.csv'):
    labels_df = pd.read_csv(BASE_DIR+'/data/labels.csv')
    extractor = FeatureExtractor()

    all_data = []

    print("[!] Processing images...")
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        image_path = row['image_path']
        gender = row['gender']

        feature_df = extractor.transform(image_path, label=gender)
        all_data.append(feature_df)

    full_df = pd.concat(all_data, ignore_index=True)

    full_df.to_csv(output_raw, index=False)
    print(f"[!] Processed data saved to: {output_raw}")