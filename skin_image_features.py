"""Shared image preprocessing and PyTorch model definitions for skin screening."""
from pathlib import Path


def create_skin_transform():
    """Return the official ImageNet preprocessing used by MobileNetV2 weights."""
    from torchvision.models import MobileNet_V2_Weights

    return MobileNet_V2_Weights.DEFAULT.transforms()


def create_pretrained_skin_extractor():
    """Build an ImageNet-pretrained MobileNetV2 that returns 1280-value embeddings.

    Torchvision raises when the official checkpoint cannot be downloaded. That
    failure is intentionally surfaced to the caller; random weights must never
    be presented as pretrained weights.
    """
    import torch
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    try:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    except Exception as exc:
        raise RuntimeError(
            "Pretrained MobileNetV2 ImageNet weights are unavailable. Connect to "
            "the internet once and run `python train_skin_model.py`. To cache the "
            "official checkpoint first, run `python -c \"from torchvision.models "
            "import MobileNet_V2_Weights; MobileNet_V2_Weights.DEFAULT.get_state_dict(" 
            "progress=True, check_hash=True)\"`. No untrained fallback was used."
        ) from exc
    model.classifier = torch.nn.Identity()
    model.eval()
    return model


def create_skin_classifier(model_name, class_count, pretrained=False):
    """Create one of the two PyTorch disease classifiers used for skin images."""
    import torch
    from torch import nn
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    if model_name == 'custom_cnn':
        class CustomSkinCNN(nn.Module):
            def __init__(self, output_count):
                super().__init__()
                self.features = nn.Sequential(
                    self._block(3, 32),
                    self._block(32, 64),
                    self._block(64, 128),
                    self._block(128, 256),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(), nn.Dropout(p=0.35), nn.Linear(256, output_count)
                )

            @staticmethod
            def _block(input_channels, output_channels):
                return nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                )

            def forward(self, image):
                return self.classifier(self.features(image))

        return CustomSkinCNN(class_count)

    if model_name == 'mobilenet_v2':
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        try:
            model = mobilenet_v2(weights=weights)
        except Exception as exc:
            raise RuntimeError(
                "Pretrained MobileNetV2 ImageNet weights are unavailable. "
                "Run `python -c \"from torchvision.models import MobileNet_V2_Weights; "
                "MobileNet_V2_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)\"` "
                "to cache the official checkpoint and retry; "
                "random weights are not used as a substitute."
            ) from exc
        model.classifier[-1] = nn.Linear(model.last_channel, class_count)
        return model

    raise ValueError(f'Unsupported skin neural model: {model_name}')


def load_skin_extractor(weights_path, device='cpu'):
    """Load the MobileNetV2 embedding backbone saved during skin training."""
    import torch
    from torchvision.models import mobilenet_v2

    model = mobilenet_v2(weights=None)
    model.classifier = torch.nn.Identity()
    state = torch.load(Path(weights_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_skin_classifier(weights_path, model_name, class_count, device='cpu'):
    """Load trained Custom CNN or MobileNetV2 classifier weights."""
    import torch

    model = create_skin_classifier(model_name, class_count, pretrained=False)
    checkpoint = torch.load(Path(weights_path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise RuntimeError(f'Trained weights for {model_name} have an invalid format.')
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint.get('class_names')


def extract_skin_features(image, model, transform, device='cpu'):
    """Return one 1280-value MobileNetV2 embedding for a PIL image."""
    import torch

    tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.inference_mode():
        return model(tensor).squeeze(0).cpu().numpy()
