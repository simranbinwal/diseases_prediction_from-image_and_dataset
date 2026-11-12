"""Train and evaluate the four skin image models on the existing dataset.

The dataset must use datasets/SkinDisease/{train,test}/<class>/*.jpg (PNG and
other common image formats are also supported). MobileNetV2 uses Torchvision's
official ImageNet weights. If they are not cached, training needs an internet
connection once; alternatively download MobileNet_V2_Weights.DEFAULT with
Torchvision and place its official checkpoint in the Torch hub checkpoints
directory. The script fails clearly instead of substituting random weights.
"""
import argparse
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from skin_image_features import (create_pretrained_skin_extractor,
                                 create_skin_classifier, create_skin_transform)


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / 'datasets' / 'SkinDisease'
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
SEED = 42
MODEL_NAMES = ('logistic_regression', 'custom_cnn', 'mobilenet_v2', 'random_forest')


class SkinImageDataset(Dataset):
    """Open the validated image records lazily so training stays memory bounded."""
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label, _image_key = self.records[index]
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert('RGB')
        return self.transform(image), int(label)


def image_files(folder):
    return sorted(path for path in folder.iterdir() if path.is_file()
                  and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def collect_split(split, class_names):
    """Validate images and build stable (path, integer-label, digest) records."""
    records = []
    for label, class_name in enumerate(class_names):
        class_dir = DATASET / split / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f'Missing skin dataset class folder: {class_dir}')
        paths = image_files(class_dir)
        if not paths:
            raise ValueError(f'No supported images found in {class_dir}')
        loaded = 0
        for path in paths:
            try:
                with Image.open(path) as opened:
                    image = ImageOps.exif_transpose(opened).convert('RGB')
                    image.load()
                    resampling = getattr(Image, 'Resampling', Image).BILINEAR
                    canonical = ImageOps.fit(image, (64, 64), method=resampling)
                    digest = sha256(np.asarray(canonical, dtype=np.uint8).tobytes()).hexdigest()
                records.append((path, label, digest))
                loaded += 1
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                print(f'Skipping unreadable image {path}: {exc}')
        if not loaded:
            raise ValueError(f'No readable images found in {class_dir}')
        print(f'{split}: validated {class_name} ({loaded} images)')
    return records


def remove_duplicate_images(train_records, test_records):
    """Remove exact duplicates and exclude any duplicate with conflicting labels."""
    labels_by_image = defaultdict(set)
    for _path, label, digest in train_records + test_records:
        labels_by_image[digest].add(label)
    ambiguous = {digest for digest, labels in labels_by_image.items() if len(labels) > 1}

    kept_train, seen_train = [], set()
    ambiguous_train = duplicate_train = 0
    for record in train_records:
        _path, _label, digest = record
        if digest in ambiguous:
            ambiguous_train += 1
        elif digest in seen_train:
            duplicate_train += 1
        else:
            kept_train.append(record)
            seen_train.add(digest)

    kept_test, seen_test = [], set()
    ambiguous_test = duplicate_test = 0
    for record in test_records:
        _path, _label, digest = record
        if digest in ambiguous:
            ambiguous_test += 1
        elif digest in seen_train or digest in seen_test:
            duplicate_test += 1
        else:
            kept_test.append(record)
            seen_test.add(digest)

    return kept_train, kept_test, {
        'train_duplicates_removed': duplicate_train,
        'test_duplicates_removed': duplicate_test,
        'ambiguous_train_images_removed': ambiguous_train,
        'ambiguous_test_images_removed': ambiguous_test,
    }


def make_loader(records, transform, batch_size, shuffle=False):
    return DataLoader(SkinImageDataset(records, transform), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def extract_embeddings(records, extractor, transform, device, batch_size):
    loader = make_loader(records, transform, batch_size)
    embeddings, labels = [], []
    with torch.inference_mode():
        for images, targets in loader:
            vectors = extractor(images.to(device, non_blocking=True))
            embeddings.append(vectors.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(embeddings).astype(np.float32), np.concatenate(labels).astype(np.int64)


def make_metrics(y_true, y_pred, class_names, evaluation_method, train_count,
                 test_count, duplicate_summary, feature_extractor=None):
    result = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, labels=range(len(class_names)),
                                           average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, labels=range(len(class_names)),
                                     average='weighted', zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, labels=range(len(class_names)),
                             average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred,
                                             labels=range(len(class_names))).tolist(),
        'labels': class_names,
        'evaluation_method': evaluation_method,
        'train_images': int(train_count),
        'test_images': int(test_count),
        'class_count': len(class_names),
        **{key: int(value) for key, value in duplicate_summary.items()},
    }
    if feature_extractor:
        result['feature_extractor'] = feature_extractor
    return result


def train_neural_classifier(model, model_name, train_records, class_names,
                            transform, device, epochs, batch_size, save_path):
    model.to(device)
    if model_name == 'mobilenet_v2':
        # Keep the pretrained visual backbone fixed; train the disease head only.
        model.features.requires_grad_(False)
        model.features.eval()
        trainable_parameters = model.classifier.parameters()
        learning_rate = 0.002
    else:
        trainable_parameters = model.parameters()
        learning_rate = 0.001

    class_counts = np.bincount([record[1] for record in train_records], minlength=len(class_names))
    class_weights = len(train_records) / np.maximum(class_counts, 1) / len(class_names)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=1e-4)
    loader = make_loader(train_records, transform, batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        if model_name == 'mobilenet_v2':
            model.features.eval()
        total_loss, seen = 0.0, 0
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            seen += len(targets)
        print(f'{model_name}: epoch {epoch + 1}/{epochs}, loss={total_loss / max(seen, 1):.4f}')

    model.eval()
    torch.save({
        'state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
        'class_names': class_names,
        'model_name': model_name,
    }, save_path)
    return model


def predict_neural_classifier(model, records, transform, device, batch_size):
    loader = make_loader(records, transform, batch_size)
    predictions, labels = [], []
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device, non_blocking=True))
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            labels.extend(targets.numpy().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(predictions, dtype=np.int64)


def main():
    global DATASET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATASET,
                        help='Dataset root containing train/ and test/ class folders.')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs for the Custom CNN and MobileNetV2 classifier.')
    parser.add_argument('--batch-size', type=int, default=48)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('--epochs and --batch-size must be positive integers.')

    DATASET = args.dataset.resolve()
    train_root, test_root = DATASET / 'train', DATASET / 'test'
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError(
            f'Skin dataset requires both train/ and test/ folders under {DATASET}.'
        )
    class_names = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    test_classes = sorted(path.name for path in test_root.iterdir() if path.is_dir())
    if len(class_names) < 2:
        raise ValueError('Skin dataset must contain at least two class folders in train/.')
    if test_classes != class_names:
        raise ValueError('Skin train/ and test/ must contain the same disease class folders.')

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}; classes={len(class_names)}; pretrained MobileNetV2 ImageNet backbone.')

    # Loading these weights is required for both embedding classifiers and MobileNetV2.
    extractor = create_pretrained_skin_extractor().to(device)
    transform = create_skin_transform()
    train_records = collect_split('train', class_names)
    test_records = collect_split('test', class_names)
    train_records, test_records, duplicate_summary = remove_duplicate_images(
        train_records, test_records
    )
    if not train_records or not test_records:
        raise ValueError('Train and test sets must both contain readable images after duplicate filtering.')
    print(f'After duplicate filtering: {len(train_records)} train, {len(test_records)} test; '
          f'{duplicate_summary}.')

    model_dir = ROOT / 'models'
    result_dir = ROOT / 'model_results'
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    with (model_dir / 'skin_class_names.json').open('w', encoding='utf-8') as handle:
        json.dump(class_names, handle, indent=2)
    extractor.cpu()
    torch.save(extractor.state_dict(), model_dir / 'skin_mobilenetv2_features.pth')
    extractor.to(device).eval()

    print('Extracting pretrained MobileNetV2 image embeddings for Logistic Regression and Random Forest...')
    X_train, y_train = extract_embeddings(train_records, extractor, transform, device, args.batch_size)
    X_test, y_test = extract_embeddings(test_records, extractor, transform, device, args.batch_size)
    base_evaluation = 'official held-out test folder; ImageNet-pretrained MobileNetV2 embeddings; exact duplicates removed'
    results = {}

    logistic = Pipeline([
        ('scale', StandardScaler()),
        ('classifier', LogisticRegression(
            C=0.5, max_iter=700, class_weight='balanced', solver='lbfgs',
            random_state=SEED,
        )),
    ])
    print(f'Training Logistic Regression on {len(y_train)} image embeddings...')
    logistic.fit(X_train, y_train)
    logistic_pred = logistic.predict(X_test)
    results['logistic_regression'] = make_metrics(
        y_test, logistic_pred, class_names, base_evaluation,
        len(train_records), len(test_records), duplicate_summary,
        'MobileNetV2 pretrained on ImageNet, frozen 1280-value image embeddings',
    )
    joblib.dump(logistic, model_dir / 'skin_logistic_regression.pkl')

    forest = RandomForestClassifier(
        n_estimators=120, max_depth=24, min_samples_leaf=2,
        max_features='sqrt', class_weight='balanced_subsample',
        n_jobs=-1, random_state=SEED,
    )
    print(f'Training Random Forest on {len(y_train)} image embeddings...')
    forest.fit(X_train, y_train)
    forest_pred = forest.predict(X_test)
    results['random_forest'] = make_metrics(
        y_test, forest_pred, class_names, base_evaluation,
        len(train_records), len(test_records), duplicate_summary,
        'MobileNetV2 pretrained on ImageNet, frozen 1280-value image embeddings',
    )
    joblib.dump(forest, model_dir / 'skin_random_forest.pkl')

    # Augment only the training view; image size, channels, and ImageNet
    # normalization stay aligned with the official inference preprocessing.
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transform,
    ])

    custom_cnn = create_skin_classifier('custom_cnn', len(class_names))
    custom_cnn = train_neural_classifier(
        custom_cnn, 'custom_cnn', train_records, class_names, train_transform,
        device, args.epochs, args.batch_size, model_dir / 'skin_custom_cnn.pth',
    )
    cnn_y, cnn_pred = predict_neural_classifier(
        custom_cnn, test_records, transform, device, args.batch_size
    )
    results['custom_cnn'] = make_metrics(
        cnn_y, cnn_pred, class_names, base_evaluation,
        len(train_records), len(test_records), duplicate_summary,
        'Custom CNN trained directly on RGB skin images',
    )

    mobile_net = create_skin_classifier('mobilenet_v2', len(class_names), pretrained=True)
    mobile_net = train_neural_classifier(
        mobile_net, 'mobilenet_v2', train_records, class_names, train_transform,
        device, args.epochs, args.batch_size, model_dir / 'skin_mobilenetv2_classifier.pth',
    )
    mobile_y, mobile_pred = predict_neural_classifier(
        mobile_net, test_records, transform, device, args.batch_size
    )
    results['mobilenet_v2'] = make_metrics(
        mobile_y, mobile_pred, class_names, base_evaluation,
        len(train_records), len(test_records), duplicate_summary,
        'MobileNetV2 ImageNet pretrained backbone; frozen backbone with a trained disease-classification head',
    )

    # Keep display order equal to the skin model selector.
    results = {name: results[name] for name in MODEL_NAMES}
    (result_dir / 'skin_model_metrics.json').write_text(
        json.dumps(results, indent=2), encoding='utf-8'
    )
    for name in MODEL_NAMES:
        print(f"{name}: held_out_test_accuracy={results[name]['accuracy']:.3f}, "
              f"weighted_f1={results[name]['f1']:.3f}")
    print(f'Saved all four skin models, class mapping, and held-out metrics to {model_dir}.')


if __name__ == '__main__':
    main()
