import joblib

def save(path, model):
    joblib.dump(model, path)
    print(f'[!] Model was Saved as {path}')

def load(path):
    return joblib.load(path)