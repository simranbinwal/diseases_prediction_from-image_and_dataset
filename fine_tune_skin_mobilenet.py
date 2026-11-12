"""Fine-tune the skin MobileNetV2 classifier and evaluate a safe candidate.

The existing skin test folder remains held out. Training uses a stratified
validation split from the training folder, and candidate files are written to
models/skin_candidates so the app's current model is not replaced automatically.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torchvision import transforms

import train_skin_model as skin_training
from skin_image_features import (create_skin_classifier,
                                 create_skin_transform)


ROOT = Path(__file__).resolve().parent


def run_epoch(model, loader, criterion, device, optimizer=None, fine_tune=False):
    training = optimizer is not None
    model.train(training)
    if training:
        model.features.eval()
        if fine_tune:
            for block in model.features[-4:]:
                block.train()

    losses = []
    true_labels, predicted_labels = [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach()) * len(targets))
            true_labels.extend(targets.cpu().tolist())
            predicted_labels.extend(logits.argmax(dim=1).cpu().tolist())

    count = max(len(true_labels), 1)
    return {
        'loss': sum(losses) / count,
        'accuracy': float(accuracy_score(true_labels, predicted_labels)),
        'macro_f1': float(f1_score(true_labels, predicted_labels,
                                   average='macro', zero_division=0)),
    }


def make_optimizer(model, fine_tune, backbone_lr, head_lr):
    if not fine_tune:
        return torch.optim.AdamW(model.classifier.parameters(), lr=head_lr,
                                 weight_decay=1e-4)
    return torch.optim.AdamW([
        {'params': model.features[-4:].parameters(), 'lr': backbone_lr},
        {'params': model.classifier.parameters(), 'lr': head_lr},
    ], weight_decay=1e-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path,
                        default=ROOT / 'datasets' / 'SkinDisease')
    parser.add_argument('--epochs', type=int, default=8,
                        help='Maximum epochs including the classifier warm-up.')
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'models' / 'skin_candidates')
    parser.add_argument('--metrics-file', type=Path,
                        default=ROOT / 'model_results' / 'skin_mobilenetv2_candidate_metrics.json')
    args = parser.parse_args()
    if args.epochs < 3 or args.batch_size < 1 or args.patience < 1:
        parser.error('--epochs must be at least 3; batch size and patience must be positive.')

    skin_training.DATASET = args.dataset.resolve()
    train_root = skin_training.DATASET / 'train'
    test_root = skin_training.DATASET / 'test'
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError('Skin dataset must contain train/ and test/ folders.')
    class_names = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    if len(class_names) < 2 or sorted(path.name for path in test_root.iterdir()
                                      if path.is_dir()) != class_names:
        raise ValueError('Train and test folders must contain the same two or more classes.')

    torch.manual_seed(skin_training.SEED)
    np.random.seed(skin_training.SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}; classes={len(class_names)}; validation split=10%.')

    train_records = skin_training.collect_split('train', class_names)
    test_records = skin_training.collect_split('test', class_names)
    train_records, test_records, duplicate_summary = skin_training.remove_duplicate_images(
        train_records, test_records
    )
    labels = np.asarray([record[1] for record in train_records])
    fit_records, validation_records = train_test_split(
        train_records, test_size=0.10, random_state=skin_training.SEED,
        stratify=labels,
    )
    print(f'Fit images={len(fit_records)}; validation images={len(validation_records)}; '
          f'held-out test images={len(test_records)}.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'skin_class_names.json').open('w', encoding='utf-8') as handle:
        json.dump(class_names, handle, indent=2)

    base_transform = create_skin_transform()
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        base_transform,
    ])
    train_loader = skin_training.make_loader(
        fit_records, train_transform, args.batch_size, shuffle=True
    )
    validation_loader = skin_training.make_loader(
        validation_records, base_transform, args.batch_size
    )
    test_loader = skin_training.make_loader(test_records, base_transform, args.batch_size)

    model = create_skin_classifier('mobilenet_v2', len(class_names), pretrained=True)
    model.to(device)
    for parameter in model.features.parameters():
        parameter.requires_grad_(False)

    counts = np.bincount([record[1] for record in fit_records], minlength=len(class_names))
    weights = len(fit_records) / np.maximum(counts, 1) / len(class_names)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )

    warmup_epochs = 2
    best_accuracy = -1.0
    best_state = None
    stale_epochs = 0
    fine_tune = False
    optimizer = make_optimizer(model, fine_tune, 1e-5, 1e-3)

    for epoch in range(args.epochs):
        if epoch == warmup_epochs:
            fine_tune = True
            for block in model.features[-4:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            optimizer = make_optimizer(model, fine_tune, 1e-5, 2e-4)
            print('Fine-tuning the final four MobileNetV2 feature blocks.')

        training_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer,
            fine_tune=fine_tune,
        )
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        print(
            f"epoch {epoch + 1}/{args.epochs}: "
            f"train_acc={training_metrics['accuracy']:.4f}, "
            f"val_acc={validation_metrics['accuracy']:.4f}, "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f}"
        )

        if validation_metrics['accuracy'] > best_accuracy:
            best_accuracy = validation_metrics['accuracy']
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            torch.save({
                'state_dict': best_state,
                'class_names': class_names,
                'model_name': 'mobilenet_v2',
                'training_method': 'ImageNet initialization; classifier warm-up; final four feature blocks fine-tuned',
                'validation_accuracy': best_accuracy,
            }, args.output_dir / 'skin_mobilenetv2_classifier.pth')
            stale_epochs = 0
        else:
            stale_epochs += 1
            if fine_tune and stale_epochs >= args.patience:
                print('Early stopping after validation accuracy stopped improving.')
                break

    if best_state is None:
        raise RuntimeError('Training did not produce a validation checkpoint.')
    model.load_state_dict(best_state)
    model.to(device).eval()

    test_metrics = run_epoch(model, test_loader, criterion, device)
    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    result = skin_training.make_metrics(
        np.asarray([record[1] for record in test_records]),
        _predict(model, test_loader, device), class_names,
        'official held-out test folder; exact duplicates removed; model selected using a stratified validation split from train',
        len(fit_records), len(test_records), duplicate_summary,
        'ImageNet-pretrained MobileNetV2; classifier warm-up followed by fine-tuning of the final four feature blocks',
    )
    result['validation_accuracy'] = best_accuracy
    result['test_macro_f1'] = test_metrics['macro_f1']
    result['test_accuracy'] = test_metrics['accuracy']
    result['training_method'] = 'fine-tuned final four MobileNetV2 feature blocks; best validation-accuracy checkpoint'
    args.metrics_file.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"Candidate held-out test accuracy={result['accuracy']:.4f}; "
          f"weighted_f1={result['f1']:.4f}; macro_f1={result['test_macro_f1']:.4f}.")
    print(f'Candidate weights: {args.output_dir / "skin_mobilenetv2_classifier.pth"}')
    print(f'Candidate metrics: {args.metrics_file}')


def _predict(model, loader, device):
    predictions = []
    model.eval()
    with torch.inference_mode():
        for images, _targets in loader:
            logits = model(images.to(device, non_blocking=True))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return np.asarray(predictions, dtype=np.int64)


if __name__ == '__main__':
    main()
