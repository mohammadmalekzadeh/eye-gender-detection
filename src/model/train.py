import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.utlis import BASE_DIR
from src.model.save_load import save

def design_train_save_model() -> None:
    df = pd.read_csv(BASE_DIR + '/data/processed/train_df.csv')
    X = df.drop(columns='gender')
    y = df['gender'].map({'male': 1, 'female': 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss"),
        "LightGBM": LGBMClassifier()
        }

    results = []

    for name, model in models.items():
        print(f"[!] Training: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC": roc,
            "Confusion Matrix": cm.tolist()
        })

        save('_'.join(name.lower().split()), model)
        print(f'[!] {name} was trained and saved')


    results_df = pd.DataFrame(results)
    results_df.to_csv(BASE_DIR+"/outputs/reports/model_evaluation_results.csv", index=False)

    print(f'[!] Models scores saved')
