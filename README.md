# 👁 Eye Gender Detection

## 🔍 Introduction

The goal of this project is to build a gender classification model using only cropped images of eyes. By applying image processing and feature extraction techniques (like HOG), the project converts visual data into structured numerical representations suitable for classical machine learning algorithms.

---

## 🧰 Features & Data

### 🔹 Feature Set:
- Images are transformed into feature vectors using:
  - HOG (Histogram of Oriented Gradients) for capturing edge orientation and texture information.
  - (Other descriptors like LBP or raw pixel intensities may be explored later.)

### 🔹 Data Preprocessing:
- Input images are pre-cropped eye images.
- Each image is resized to a standard resolution (e.g., 64×64).
- Images are converted to grayscale.
- Feature vectors are extracted using image descriptors.
- Gender labels are read from a CSV file and aligned with feature vectors.

### 🔹 Model:
- The model is implemented using Scikit-learn.
- Initial experiments use:
  - Support Vector Machine (SVM) with RBF or linear kernels.
- Additional models can be tested (e.g., Random Forest, Logistic Regression).
- Training involves splitting the data into training/testing sets, model fitting, and evaluation.

### 🔹 Outputs:
- Trained model file saved as .pkl.
- Evaluation reports: accuracy, F1-score, Recall and etc.
- Visual performance metrics and plots.
- An interactive Jupyter Notebook (eye_gender_pipeline.ipynb) containing the full pipeline.

## 📌 How to Run the Program
  1. **Clone the Repository**
  ```bash
      git clone https://github.com/mohammadmalekzadeh/eye-gender-detection.git
      cd eye-gender-detection
  ```

  2. **Install Dependencies**
  ```python
      pip install -r requirements.txt
  ```

  3. **Run the Program**
  ```python
      python main.py
  ```

  _Finally, you can see the model results and make **prediction** on the test data_
  _Share your suggestions with us_

## 🚀 Technologies Used
```bash

    - NumPy
    - Pandas
    - Pillow
    - Scikit-Image
    - Scikit-learn
    - Joblib
    - Matplotlib
    - Seaborn
    - Jupyter Notebook

```


## 📁 Project Structure
```
salary-prediction/
├── data/
│   ├── raw/
│   ├── images/
│   ├── true_values/
│   ├── labels.csv
│   └── processed/
├── src/
│   ├── model/
│      ├── evaluate.py
│      ├── save_load.py
│      └── train.py
│   ├── preprocessing/
│      ├── label_creator.py
│      └── extra_feature.py
│   └── utlis.py
├── models/
├── outputs/
│   ├── reports/
│   └── models/
├── eye_gender_pipeline.ipynb
├── main.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```


## 📜 License
This project is licensed under the [MIT License](LICENSE).

## 📬 Contact
Maintained by **Mohammad Malekzadeh**.  
Questions? Issues? Feature requests? Just open an issue or reach out via GitHub!
