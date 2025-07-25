import joblib
from src.utlis import BASE_DIR

def save(name, model):
    joblib.dump(model, BASE_DIR+'/models/'+name+'.pkl')

def load(full_path):
    return joblib.load(full_path)