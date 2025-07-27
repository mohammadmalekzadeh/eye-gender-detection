import pandas as pd
import numpy as np
from src.utlis import BASE_DIR
from src.model.save_load import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def prediction() -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR+'/data/processed/test_df.csv')

    model = load(BASE_DIR+'/outputs/models/best_model_SVM.pkl')

    y_pred = model.predict(df)

    prediction_df = pd.DataFrame({'gender': y_pred})

    prediction_df = prediction_df['gender'].map({1: 'male', 0: 'female'})
    
    prediction_df.to_csv(BASE_DIR+'/outputs/prediction/prediction.csv', index=False)

    print('[!] Prediction dataframe Saved')
