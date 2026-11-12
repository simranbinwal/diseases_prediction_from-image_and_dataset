# Multi-Disease Risk Prediction

A Flask web application for educational screening across heart disease risk, mental stress, and skin image classification. The interface accepts screening inputs, runs a selected trained model, displays the result and general recommendations, and can generate a PDF report.

> **Disclaimer:** This project is for educational screening only. Its results are not a diagnosis and do not replace advice or care from a qualified health professional.

## Features

- Heart screening using Logistic Regression, Decision Tree, Random Forest, or SVM.
- Stress-level screening using Logistic Regression, Decision Tree, Random Forest, or SVM.
- Skin image screening using trained image classification models.
- Model performance summaries and confusion matrices in the web interface.
- Input validation, SQLite storage for screenings, and downloadable PDF reports.

## Technology

Python, Flask, scikit-learn, pandas, NumPy, PyTorch, Torchvision, Pillow, SQLite, HTML, CSS, JavaScript, and ReportLab.

## Run the application

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5001> in a browser.

## Models and Git LFS

Pre-trained model artifacts and evaluation metrics are stored in `models/` and `model_results/`. The `models/skin_random_forest.pkl` artifact is stored with Git LFS; install Git LFS before cloning so the model file is downloaded correctly.

## Training data

The training datasets are not included in this repository. They contain health-related records and/or a large image collection, so obtain them from their original source and follow that source's terms before sharing or redistributing them.

Expected paths:

- `datasets/heart.csv`
- `datasets/stress.csv`
- Skin images arranged as `datasets/SkinDisease/{train,test}/<class-name>/<image-file>`

After placing the data locally, run the corresponding training script:

```bash
python train_heart_model.py
python train_stress_model.py
python train_skin_model.py
```

`train_skin_model.py` also accepts a custom dataset root with `--dataset PATH`. Skin model training may need to download pretrained Torchvision weights the first time.

## Project structure

- `app.py` — Flask routes, prediction handling, validation, SQLite persistence, and PDF reports.
- `templates/` and `static/` — web page and styles.
- `train_*_model.py` — model training and evaluation scripts.
- `skin_image_features.py` and `fine_tune_skin_mobilenet.py` — skin image model utilities and fine-tuning.
- `models/` — saved model artifacts.
- `model_results/` — evaluation metrics displayed by the app.
