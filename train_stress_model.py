"""Train and evaluate stress score classifiers without duplicate-profile leakage."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
FEATURES = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
            'Heart Rate', 'Daily Steps']
TARGET = 'Stress Level'
OUTER_FOLDS = 5
INNER_FOLDS = 3
SEED = 42


def model_specs():
    """Use nested tuning with restricted tree depth and leaf size."""
    return {
        'logistic_regression': (
            Pipeline([('scaler', StandardScaler()),
                      ('model', LogisticRegression(max_iter=3000, random_state=SEED))]),
            {'model__C': [0.1, 1.0, 10.0]},
        ),
        'decision_tree': (
            DecisionTreeClassifier(random_state=SEED),
            {'max_depth': [4, 5], 'min_samples_leaf': [4, 6, 8]},
        ),
        'random_forest': (
            RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
            {'max_depth': [2, 3, 4], 'min_samples_leaf': [4, 6, 8]},
        ),
        'svm': (
            Pipeline([('scaler', StandardScaler()),
                      ('model', SVC(C=1.0, kernel='rbf',
                                    decision_function_shape='ovr', random_state=SEED))]),
            {},
        ),
    }


def unique_profiles(data):
    """Collapse repeated feature vectors, refusing inconsistent duplicate labels."""
    labels_per_profile = data.groupby(FEATURES, dropna=False)[TARGET].nunique()
    if (labels_per_profile > 1).any():
        raise ValueError('Identical stress input profiles have conflicting target labels.')
    return data.drop_duplicates(subset=FEATURES, keep='first').reset_index(drop=True)


def auc_for_fold(estimator, X_test, y_test, labels):
    try:
        if hasattr(estimator, 'predict_proba'):
            scores = estimator.predict_proba(X_test)
        else:
            scores = estimator.decision_function(X_test)
        if len(labels) == 2:
            if getattr(scores, 'ndim', 1) == 2:
                classes = list(estimator.classes_)
                scores = scores[:, classes.index(labels[1])]
            return float(roc_auc_score(y_test, scores))
        encoded_labels = label_binarize(y_test, classes=labels)
        return float(roc_auc_score(encoded_labels, scores, average='macro'))
    except ValueError:
        return None


def nested_metrics(estimator, parameter_grid, X, y, labels):
    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + 1)
    y_true, y_pred, fold_accuracies, fold_aucs = [], [], [], []

    for train_idx, test_idx in outer_cv.split(X, y):
        search = GridSearchCV(estimator, parameter_grid, scoring='accuracy',
                              cv=inner_cv, n_jobs=1, refit=True)
        search.fit(X.iloc[train_idx], y.iloc[train_idx])
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        predictions = search.predict(X_test)
        y_true.extend(y_test.tolist())
        y_pred.extend(predictions.tolist())
        fold_accuracies.append(accuracy_score(y_test, predictions))
        auc = auc_for_fold(search.best_estimator_, X_test, y_test, labels)
        if auc is not None:
            fold_aucs.append(auc)

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'accuracy_std': float(np.std(fold_accuracies, ddof=1)),
        'fold_accuracies': [float(score) for score in fold_accuracies],
        'precision': float(precision_score(y_true, y_pred, average='weighted',
                                           zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted',
                                     zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'labels': labels,
        'roc_auc': float(np.mean(fold_aucs)) if fold_aucs else None,
        'evaluation_method': 'nested stratified 5-fold CV on unique input profiles',
    }


def main():
    raw_data = pd.read_csv(ROOT / 'datasets' / 'stress.csv')
    data = unique_profiles(raw_data)
    X, y = data[FEATURES], data[TARGET].astype(int)
    labels = sorted(y.unique().tolist())
    model_dir = ROOT / 'models'
    result_dir = ROOT / 'model_results'
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    results = {}
    for name, (estimator, parameter_grid) in model_specs().items():
        row = nested_metrics(estimator, parameter_grid, X, y, labels)
        row.update({
            'unique_profiles': int(len(data)),
            'raw_rows': int(len(raw_data)),
            'duplicate_profiles_removed': int(len(raw_data) - len(data)),
        })

        final_search = GridSearchCV(
            estimator, parameter_grid, scoring='accuracy',
            cv=StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED),
            n_jobs=1, refit=True,
        )
        final_search.fit(X, y)
        row['best_parameters'] = final_search.best_params_
        results[name] = row
        joblib.dump(final_search.best_estimator_, model_dir / f'stress_{name}.pkl')
        print(f"{name}: nested_cv_accuracy={row['accuracy']:.3f} "
              f"(+/- {row['accuracy_std']:.3f}), ROC-AUC={row['roc_auc']}")

    (result_dir / 'stress_model_metrics.json').write_text(
        json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
