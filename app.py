from datetime import datetime
from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import Path
import secrets
import sqlite3

from flask import Flask, jsonify, render_template, request, send_file
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename

from skin_image_features import (create_skin_transform, extract_skin_features,
                                 load_skin_classifier, load_skin_extractor)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 9 * 1024 * 1024
ROOT = Path(__file__).resolve().parent
DATABASE_PATH = ROOT / 'instance' / 'screenings.sqlite3'
MAX_SKIN_IMAGE_PIXELS = 16_000_000
MODEL_NAMES = ('logistic_regression', 'decision_tree', 'random_forest', 'svm')
MODEL_LABELS = {'logistic_regression': 'Logistic Regression', 'decision_tree': 'Decision Tree', 'random_forest': 'Random Forest', 'svm': 'SVM'}
SKIN_MODEL_NAMES = ('logistic_regression', 'custom_cnn', 'mobilenet_v2', 'random_forest')
SKIN_MODEL_LABELS = {'logistic_regression': 'Logistic Regression', 'custom_cnn': 'Custom CNN',
                     'mobilenet_v2': 'MobileNetV2', 'random_forest': 'Random Forest'}
ALL_MODEL_LABELS = {**MODEL_LABELS, **SKIN_MODEL_LABELS}
FEATURES = {
    'heart': ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang'],
    'stress': ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Heart Rate', 'Daily Steps'],
}
DISPLAY_FIELDS = {
    'age': 'Age', 'sex': 'Sex code', 'cp': 'Chest pain type code', 'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Cholesterol (mg/dl)', 'thalach': 'Maximum heart rate', 'exang': 'Exercise induced angina code',
    'sleep_duration': 'Sleep duration (hours)', 'sleep_quality': 'Quality of sleep',
    'physical_activity': 'Physical activity level (dataset units)', 'heart_rate': 'Heart rate (bpm)', 'steps': 'Daily steps',
}
DISCLAIMER = 'This tool is for educational screening only and is not a diagnosis or a substitute for advice, diagnosis, or treatment from a qualified health professional.'

@contextmanager
def database_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

def initialize_database():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with database_connection() as connection:
        connection.execute('''
            CREATE TABLE IF NOT EXISTS screenings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_token TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                user_name TEXT NOT NULL,
                disease TEXT NOT NULL,
                model TEXT NOT NULL,
                result TEXT NOT NULL,
                score INTEGER,
                probability REAL,
                inputs_json TEXT NOT NULL,
                recommendations_json TEXT NOT NULL,
                report_pdf BLOB,
                image_blob BLOB,
                image_name TEXT,
                image_mime TEXT
            )
        ''')
        existing_columns = {row[1] for row in connection.execute('PRAGMA table_info(screenings)')}
        for column, column_type in (('image_blob', 'BLOB'), ('image_name', 'TEXT'),
                                    ('image_mime', 'TEXT')):
            if column not in existing_columns:
                connection.execute(f'ALTER TABLE screenings ADD COLUMN {column} {column_type}')
        connection.execute('CREATE INDEX IF NOT EXISTS idx_screenings_created_at ON screenings(created_at)')

initialize_database()

def save_screening(data, disease, model_name, result, probability, inputs, recommendations,
                   score=None, image_bytes=None, image_name=None, image_mime=None):
    report_token = secrets.token_urlsafe(32)
    user_name = str(data.get('name', '')).strip()[:100]
    created_at = datetime.now().astimezone().isoformat(timespec='seconds')
    with database_connection() as connection:
        connection.execute('''
            INSERT INTO screenings
                (report_token, created_at, user_name, disease, model, result, score,
                 probability, inputs_json, recommendations_json, image_blob, image_name, image_mime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (report_token, created_at, user_name, disease, model_name, result, score,
              probability, json.dumps(inputs), json.dumps(recommendations),
              sqlite3.Binary(image_bytes) if image_bytes is not None else None,
              image_name, image_mime))
    return report_token

def load_models(disease, model_names=MODEL_NAMES):
    loaded = {}
    for name in model_names:
        path = ROOT / 'models' / f'{disease}_{name}.pkl'
        if path.is_file():
            if disease == 'skin':
                feature_path = ROOT / 'models' / 'skin_mobilenetv2_features.pth'
                if feature_path.is_file() and path.stat().st_mtime_ns < feature_path.stat().st_mtime_ns:
                    app.logger.warning('Ignoring outdated skin classifier: %s', path.name)
                    continue
            try:
                model = joblib.load(path)
                if disease == 'skin' and getattr(model, 'n_features_in_', None) != 1280:
                    app.logger.warning('Ignoring skin classifier trained on an incompatible feature size: %s', path.name)
                    continue
                loaded[name] = model
            except Exception:
                app.logger.exception('Could not load model file: %s', path.name)
    return loaded

HEART_MODELS = load_models('heart')
STRESS_MODELS = load_models('stress')
SKIN_FEATURES_READY = all((ROOT / 'models' / artifact).is_file() for artifact in (
    'skin_class_names.json', 'skin_mobilenetv2_features.pth'
))
SKIN_MODELS = (load_models('skin', ('logistic_regression', 'random_forest'))
               if SKIN_FEATURES_READY else {})
SKIN_FEATURE_EXTRACTOR = None
SKIN_IMAGE_TRANSFORM = None
SKIN_NEURAL_MODELS = {}
SKIN_CLASS_NAMES = None

def load_metrics():
    result = {}
    for disease in ('heart', 'stress', 'skin'):
        path = ROOT / 'model_results' / f'{disease}_model_metrics.json'
        if path.is_file():
            try:
                result[disease] = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                result[disease] = {}
        else:
            result[disease] = {}
        if disease == 'skin' and set(result[disease]) != set(SKIN_MODEL_NAMES):
            # Do not present metrics from the retired MobileNetV3/SVM/Decision Tree setup.
            result[disease] = {}
    return result

@app.route('/')
def home():
    return render_template('index.html', metrics=load_metrics(), model_labels=ALL_MODEL_LABELS)

def numeric_fields(data, keys):
    values = {}
    for key in keys:
        raw = data.get(key)
        if raw is None or raw == '':
            raise ValueError(f'Please provide {DISPLAY_FIELDS.get(key, key)}.')
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f'{DISPLAY_FIELDS.get(key, key)} must be a number.')
        if not np.isfinite(value):
            raise ValueError(f'{DISPLAY_FIELDS.get(key, key)} must be a valid number.')
        values[key] = value
    return values

def validate_model_name(data, allowed_names=MODEL_NAMES):
    name = data.get('model')
    if name not in allowed_names:
        raise ValueError('Choose one of the available machine learning models.')
    return name

def skin_embedding(image):
    global SKIN_FEATURE_EXTRACTOR, SKIN_IMAGE_TRANSFORM
    weights_path = ROOT / 'models' / 'skin_mobilenetv2_features.pth'
    if not weights_path.is_file():
        raise RuntimeError('The skin image feature model is missing. Run train_skin_model.py first.')
    if SKIN_FEATURE_EXTRACTOR is None:
        SKIN_FEATURE_EXTRACTOR = load_skin_extractor(weights_path)
        SKIN_IMAGE_TRANSFORM = create_skin_transform()
    return extract_skin_features(image, SKIN_FEATURE_EXTRACTOR, SKIN_IMAGE_TRANSFORM)


def skin_class_names():
    global SKIN_CLASS_NAMES
    if SKIN_CLASS_NAMES is None:
        path = ROOT / 'models' / 'skin_class_names.json'
        if not path.is_file():
            raise RuntimeError('The skin class mapping is missing. Run train_skin_model.py first.')
        try:
            SKIN_CLASS_NAMES = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError('The saved skin class mapping cannot be read. Retrain the skin models.') from exc
        if not isinstance(SKIN_CLASS_NAMES, list) or len(SKIN_CLASS_NAMES) < 2:
            raise RuntimeError('The saved skin class mapping is invalid. Retrain the skin models.')
    return SKIN_CLASS_NAMES


def skin_neural_classifier(model_name, class_names):
    if model_name not in SKIN_NEURAL_MODELS:
        path = ROOT / 'models' / f'skin_{model_name}.pth'
        if not path.is_file():
            raise RuntimeError(f'Trained weights for {SKIN_MODEL_LABELS[model_name]} are missing. Run train_skin_model.py first.')
        model, saved_classes = load_skin_classifier(path, model_name, len(class_names))
        if saved_classes != class_names:
            raise RuntimeError('The saved model labels do not match the skin class mapping. Retrain the skin models.')
        SKIN_NEURAL_MODELS[model_name] = model
    return SKIN_NEURAL_MODELS[model_name]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    if not isinstance(data, dict):
        return jsonify(error='Please submit a valid screening form.'), 400
    disease = None
    try:
        disease = data.get('disease')
        if disease not in ('heart', 'stress', 'skin'):
            raise ValueError('Choose Heart Disease, Mental Stress, or Skin Image screening.')
        model_name = validate_model_name(data, SKIN_MODEL_NAMES if disease == 'skin' else MODEL_NAMES)
        registry = {'heart': HEART_MODELS, 'stress': STRESS_MODELS, 'skin': SKIN_MODELS}[disease]
        model = registry.get(model_name)
        is_skin_neural_model = disease == 'skin' and model_name in ('custom_cnn', 'mobilenet_v2')
        if model is None and not is_skin_neural_model:
            app.logger.error('Requested model is missing: %s_%s', disease, model_name)
            if disease == 'skin':
                return jsonify(error='The selected skin model is not trained yet. Run train_skin_model.py first.'), 503
            return jsonify(error='This model is not available yet. Train the models and try again.'), 503

        if disease == 'skin':
            required_artifacts = [ROOT / 'models' / 'skin_class_names.json']
            if model_name in ('logistic_regression', 'random_forest'):
                required_artifacts.extend([
                    ROOT / 'models' / f'skin_{model_name}.pkl',
                    ROOT / 'models' / 'skin_mobilenetv2_features.pth',
                ])
            else:
                required_artifacts.append(ROOT / 'models' / f'skin_{model_name}.pth')
            if not all(path.is_file() for path in required_artifacts):
                return jsonify(error='The selected skin model is not trained yet. Run train_skin_model.py first.'), 503
            image_file = request.files.get('image')
            if image_file is None or not image_file.filename:
                raise ValueError('Choose a clear skin image to analyze.')
            image_bytes = image_file.read(8 * 1024 * 1024 + 1)
            if not image_bytes:
                raise ValueError('The selected image is empty.')
            if len(image_bytes) > 8 * 1024 * 1024:
                raise ValueError('The image must be 8 MB or smaller.')
            try:
                with Image.open(BytesIO(image_bytes)) as check_image:
                    check_image.verify()
                with Image.open(BytesIO(image_bytes)) as uploaded_image:
                    if uploaded_image.format not in ('JPEG', 'PNG', 'WEBP', 'BMP'):
                        raise ValueError('Use a JPEG, PNG, WebP, or BMP image.')
                    width, height = uploaded_image.size
                    if width < 32 or height < 32:
                        raise ValueError('Choose an image at least 32 by 32 pixels.')
                    if width * height > MAX_SKIN_IMAGE_PIXELS:
                        raise ValueError('The image dimensions are too large to process safely.')
                    image = ImageOps.exif_transpose(uploaded_image).convert('RGB')
                    mime_type = {'JPEG': 'image/jpeg', 'PNG': 'image/png',
                                 'WEBP': 'image/webp', 'BMP': 'image/bmp'}[uploaded_image.format]
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise ValueError('The uploaded file is not a readable image. Choose a JPEG, PNG, WebP, or BMP.') from exc

            class_names = skin_class_names()
            probability = None
            if model_name in ('logistic_regression', 'random_forest'):
                features = skin_embedding(image).reshape(1, -1)
                predicted_index = int(model.predict(features)[0])
                if hasattr(model, 'predict_proba'):
                    probability = float(np.max(model.predict_proba(features)[0]))
            else:
                neural_model = skin_neural_classifier(model_name, class_names)
                transform = create_skin_transform()
                import torch
                tensor = transform(image).unsqueeze(0)
                with torch.inference_mode():
                    logits = neural_model(tensor)
                    probabilities = torch.softmax(logits, dim=1)[0]
                    predicted_index = int(torch.argmax(probabilities).item())
                    probability = float(torch.max(probabilities).item())
            if predicted_index < 0 or predicted_index >= len(class_names):
                raise RuntimeError('The selected skin model returned an unknown class index.')
            class_name = class_names[predicted_index]
            result = f'Predicted skin category: {class_name.replace("_", " ")}'
            recommendations = skin_recommendations()
            image_name = secure_filename(image_file.filename)[:120] or 'skin_image'
            values = {'image_filename': image_name, 'image_width': width, 'image_height': height}
            screening_id = save_screening(
                data, disease, model_name, result, probability, values, recommendations,
                image_bytes=image_bytes, image_name=image_name, image_mime=mime_type,
            )
            return jsonify(result=result, class_name=class_name.replace('_', ' '),
                           probability=probability,
                           probability_label='Model score for the predicted category (not calibrated)',
                           model=model_name, model_label=SKIN_MODEL_LABELS[model_name],
                           recommendations=recommendations, disclaimer=DISCLAIMER,
                           screening_id=screening_id, disease=disease)

        if disease == 'heart':
            values = numeric_fields(data, FEATURES['heart'])
            for key in ('sex', 'cp', 'exang'):
                if not values[key].is_integer():
                    raise ValueError(f'{DISPLAY_FIELDS[key]} must be a valid category code.')
            if values['sex'] not in (0, 1) or values['exang'] not in (0, 1):
                raise ValueError('Sex and exercise induced angina must use 0 or 1.')
            if values['cp'] not in (0, 1, 2, 3):
                raise ValueError('Chest pain type must be a code from 0 to 3.')
            if not 1 <= values['age'] <= 120:
                raise ValueError('Age must be between 1 and 120 years.')
            if not 40 <= values['trestbps'] <= 300 or not 50 <= values['chol'] <= 1000 or not 30 <= values['thalach'] <= 300:
                raise ValueError('Enter valid heart measurements within the supported ranges.')
            X = pd.DataFrame([[values[k] for k in FEATURES['heart']]], columns=FEATURES['heart'])
            pred = int(model.predict(X)[0])
            result = 'High Risk' if pred == 1 else 'Low Risk'
            probability = None
            if hasattr(model, 'predict_proba'):
                classes = list(model.classes_)
                if 1 in classes:
                    probability = float(model.predict_proba(X)[0][classes.index(1)])
            recommendations = heart_recommendations(pred)
            screening_id = save_screening(data, disease, model_name, result, probability, values, recommendations)
            return jsonify(result=result, probability=probability,
                           probability_label='Estimated model probability' if probability is not None else None,
                           model=model_name, model_label=MODEL_LABELS[model_name],
                           recommendations=recommendations, disclaimer=DISCLAIMER,
                           screening_id=screening_id)

        keys = ['age', 'sleep_duration', 'sleep_quality', 'physical_activity', 'heart_rate', 'steps']
        values = numeric_fields(data, keys)
        if not 1 <= values['age'] <= 120:
            raise ValueError('Age must be between 1 and 120 years.')
        if not 0 <= values['sleep_duration'] <= 24 or not 0 <= values['sleep_quality'] <= 10:
            raise ValueError('Sleep duration must be 0 to 24 hours and sleep quality must be 0 to 10.')
        if values['physical_activity'] < 0 or not 20 <= values['heart_rate'] <= 250 or not 0 <= values['steps'] <= 100000:
            raise ValueError('Enter valid non-negative activity and step values and a heart rate from 20 to 250.')
        row = {
            'Age': values['age'], 'Sleep Duration': values['sleep_duration'],
            'Quality of Sleep': values['sleep_quality'], 'Physical Activity Level': values['physical_activity'],
            'Heart Rate': values['heart_rate'], 'Daily Steps': values['steps'],
        }
        X = pd.DataFrame([row], columns=FEATURES['stress'])
        score = int(round(float(model.predict(X)[0])))
        score = max(0, min(score, 10))
        probability = None
        if hasattr(model, 'predict_proba'):
            classes = list(model.classes_)
            if score in classes:
                probability = float(model.predict_proba(X)[0][classes.index(score)])
        result = f'Stress Level Score: {score} / 10'
        recommendations = stress_recommendations(score)
        screening_id = save_screening(data, disease, model_name, result, probability, values,
                                      recommendations, score=score)
        return jsonify(result=result, score=score,
                       probability=probability,
                       probability_label='Estimated model probability for this score' if probability is not None else None,
                       model=model_name, model_label=MODEL_LABELS[model_name],
                       recommendations=recommendations, disclaimer=DISCLAIMER,
                       screening_id=screening_id)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except ImportError:
        app.logger.exception('A prediction dependency is missing')
        if disease == 'skin':
            return jsonify(error='Skin prediction dependencies are missing. Install them with `python -m pip install -r requirements.txt`.'), 503
        return jsonify(error='A required prediction dependency is missing. Install the project requirements and try again.'), 503
    except Exception:
        app.logger.exception('Prediction failed')
        if disease == 'skin':
            return jsonify(error='Skin image prediction failed. Check the selected skin model files and try again.'), 500
        return jsonify(error='Prediction could not be completed. Please check the entered values and try again.'), 500

@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error='The uploaded request is too large. Skin images must be 8 MB or smaller.'), 413

def heart_recommendations(pred):
    if pred:
        return ['Arrange a review with a qualified healthcare professional to discuss these screening results.',
                'Ask a clinician about appropriate blood pressure, cholesterol, activity, and nutrition goals for you.',
                'Seek emergency care for severe chest pain, trouble breathing, fainting, or sudden weakness.']
    return ['Continue routine preventive care and discuss personal heart health goals with your clinician.',
            'Regular movement, a balanced eating pattern, avoiding tobacco, and routine blood pressure checks can support heart health.',
            'A low model risk result does not rule out disease; seek medical advice about symptoms or concerns.']

def stress_recommendations(score):
    if score >= 7:
        return ['Consider discussing ongoing stress, sleep, or mood concerns with a qualified healthcare or mental health professional.',
                'If possible, protect a regular sleep schedule and take short restorative breaks during the day.',
                'Reach out to someone you trust; seek urgent local help if you may harm yourself or someone else.']
    return ['Keep regular sleep and wake times and include restorative breaks and activity that feels manageable.',
            'If stress begins affecting daily life or persists, consider talking with a qualified professional.',
            'This score is a screening estimate and does not diagnose or rule out a mental health condition.']

def skin_recommendations():
    return ['This image result is an automated screening estimate, not a confirmed diagnosis.',
            'Arrange an assessment with a qualified dermatologist, especially if the spot is changing, bleeding, painful, or not healing.',
            'Seek urgent medical care for a rapidly spreading rash, severe swelling, fever, or breathing difficulty.']

def make_report_pdf(record):
    inputs = json.loads(record['inputs_json'])
    recommendations = json.loads(record['recommendations_json'])
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    _, height = A4
    c.setFont('Helvetica-Bold', 18)
    c.drawString(50, height - 50, 'Health Screening Report')
    c.setFont('Helvetica', 11)
    lines = [f"Patient/User Name: {record['user_name'][:100]}",
             f"Screening Type: {record['disease'].title()}",
             f"Date and Time: {record['created_at'][:19].replace('T', ' ')}",
             f"Selected ML Algorithm: {(SKIN_MODEL_LABELS if record['disease'] == 'skin' else MODEL_LABELS).get(record['model'], record['model'])}",
             f"Prediction Result: {record['result'][:120]}"]
    if record['probability'] is not None:
        score_label = 'Classifier score (not calibrated)' if record['disease'] == 'skin' else 'Estimated model probability'
        lines.append(f"{score_label}: {float(record['probability']) * 100:.1f}%")

    def pdf_text(value):
        # ReportLab's built-in Helvetica font supports WinAnsi, not arbitrary Unicode.
        return str(value).encode('cp1252', errors='replace').decode('cp1252')

    y = height - 90
    for line in lines:
        c.drawString(50, y, pdf_text(line))
        y -= 22
    y -= 8
    c.setFont('Helvetica-Bold', 12)
    c.drawString(50, y, 'Input Parameters')
    c.setFont('Helvetica', 10)
    y -= 20
    for key, value in inputs.items():
        if key in ('disease', 'model'):
            continue
        label = DISPLAY_FIELDS.get(key, key.replace('_', ' ').title())
        c.drawString(50, y, pdf_text(f'{label}: {str(value)[:90]}'))
        y -= 17
        if y < 110:
            c.showPage(); y = height - 55; c.setFont('Helvetica', 10)
    if record['image_blob']:
        if y < 250:
            c.showPage(); y = height - 55
        c.setFont('Helvetica-Bold', 11)
        c.drawString(50, y, 'Uploaded skin image')
        y -= 12
        c.drawImage(ImageReader(BytesIO(record['image_blob'])), 50, y - 150,
                    width=200, height=145, preserveAspectRatio=True, anchor='c', mask='auto')
        y -= 165
    y -= 10
    c.setFont('Helvetica-Bold', 11)
    c.drawString(50, y, 'General next steps')
    c.setFont('Helvetica', 9)
    y -= 17
    for item in recommendations[:4]:
        c.drawString(50, y, pdf_text(('- ' + str(item))[:110]))
        y -= 14
        if y < 70:
            c.showPage(); y = height - 55; c.setFont('Helvetica', 9)
    y = max(y - 2, 45)
    c.setFont('Helvetica-Oblique', 8)
    for idx in range(0, len(DISCLAIMER), 105):
        c.drawString(50, y, pdf_text(DISCLAIMER[idx:idx + 105]))
        y -= 11
    c.save()
    return buffer.getvalue()

@app.route('/download-report', methods=['POST'])
def download_report():
    data = request.get_json(silent=True)
    screening_id = data.get('screening_id') if isinstance(data, dict) else None
    if not isinstance(screening_id, str) or not screening_id:
        return jsonify(error='Run a screening first, then request its saved report.'), 400

    with database_connection() as connection:
        connection.row_factory = sqlite3.Row
        record = connection.execute(
            'SELECT * FROM screenings WHERE report_token = ?', (screening_id,)
        ).fetchone()
        if record is None:
            return jsonify(error='This screening report could not be found. Run the screening again.'), 404
        pdf_bytes = record['report_pdf']
        if pdf_bytes is None:
            pdf_bytes = make_report_pdf(record)
            connection.execute(
                'UPDATE screenings SET report_pdf = ? WHERE report_token = ?',
                (sqlite3.Binary(pdf_bytes), screening_id)
            )

    return send_file(BytesIO(pdf_bytes), mimetype='application/pdf', as_attachment=True,
                     download_name='health_report.pdf')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
