from src.utlis import BASE_DIR
from src.model.save_load import save, load
import pandas as pd
import numpy as np

def select_best_model():
    df = pd.read_csv(BASE_DIR+'/outputs/reports/model_evaluation_results.csv')
    df['AveageScore'] = df[df.columns.to_list()[1:6]].mean(axis=1)
    
    df_sorted = df.sort_values(by='AveageScore', ascending=False)

    best_model = pd.DataFrame([df_sorted.iloc[0]])

    print(f'[!] Best Model Selected Based on Aveage Metrics')
    for col in best_model.columns.to_list()[:6]:
        value = best_model[col].values[0]
        if type(value) is np.float64:
            print(f'[!] {col}\t=> {value*100:.3f}%')
        else:
            print(f'[!] {col}\t=> {value}')
            name = value

    model = load(BASE_DIR+'/models/'+'_'.join(name.lower().split())+'.pkl')
    save(name, model, best=True)
