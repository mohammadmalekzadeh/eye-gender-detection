import joblib
from src.utlis import BASE_DIR

def save(name, model, best=False):
    if best:
        joblib.dump(model, BASE_DIR+'/outputs/models/best_model_'+name+'.pkl')
    else:
        joblib.dump(model, BASE_DIR+'/models/'+name+'.pkl')

def load(full_path):
    return joblib.load(full_path)