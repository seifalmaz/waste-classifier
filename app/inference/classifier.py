import os
import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import models, transforms
import torch.nn as nn


CLASS_NAMES = [
    "Glass",
    "Metal",
    "Organic",
    "Paper",
    "Plastic"
]


class WasteClassifier:
    def __init__(
        self,
        model_path,
        model_name="efficientnet_b3",
        mode="realtime"
    ):

        self.model_path = model_path
        self.model_name = model_name
        self.mode = mode

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = self._load_model()

        self.transform = self._build_transform()

    # =========================================================
    # MODEL LOADING
    # =========================================================

    def _load_model(self):

        if self.model_name == "efficientnet_b3":

            model = models.efficientnet_b3(weights=None)

            in_features = model.classifier[1].in_features

            model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, len(CLASS_NAMES))
            )

        else:
            raise ValueError(
                f"Unsupported model: {self.model_name}"
            )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )

        if isinstance(checkpoint, dict):

            state_dict = (
                checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint
            )

        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)

        model.to(self.device)

        model.eval()

        return model

    # =========================================================
    # TRANSFORMS
    # =========================================================

    def _build_transform(self):

        if self.mode == "realtime":

            return transforms.Compose([

                transforms.ToPILImage(),

                transforms.Resize((224, 224)),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        elif self.mode == "upload":

            return transforms.Compose([

                transforms.ToPILImage(),

                transforms.Resize((300, 300)),

                transforms.CenterCrop(224),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        else:
            raise ValueError(
                f"Unsupported mode: {self.mode}"
            )

    # =========================================================
    # PREPROCESS
    # =========================================================

    def preprocess(self, image):

        if isinstance(image, str):

            image = cv2.imread(image)

            if image is None:
                raise FileNotFoundError(
                    f"Could not load image: {image}"
                )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )
        elif isinstance(image, np.ndarray):

            pass

        else:
            raise ValueError(
                "Unsupported image format"
            )

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        return tensor

    # =========================================================
    # PREDICTION
    # =========================================================

    @torch.inference_mode()
    def predict(self, image):

        tensor = self.preprocess(image)

        outputs = self.model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        confidence, predicted_idx = torch.max(
            probabilities,
            dim=0
        )

        predicted_class = CLASS_NAMES[
            predicted_idx.item()
        ]

        return {

            "class": predicted_class,

            "confidence": float(confidence.item()),

            "probabilities": {

                CLASS_NAMES[i]: float(probabilities[i].item())

                for i in range(len(CLASS_NAMES))
            }
        }