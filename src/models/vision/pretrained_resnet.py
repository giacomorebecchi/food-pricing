import logging
from typing import Optional

import torch
from torchvision.models import resnet152

logging.basicConfig(level=logging.INFO)


class PreTrainedResNet152(torch.nn.Module):
    def __init__(
        self,
        weights: Optional[str] = "DEFAULT",
        feature_dim: Optional[int] = None,
    ):
        super(PreTrainedResNet152, self).__init__()
        self.resnet = resnet152(weights="DEFAULT")
        in_features = self.resnet.fc.in_features
        self.freeze_encoder()
        if feature_dim is not None:
            self.resnet.fc = torch.nn.Linear(
                in_features=in_features, out_features=feature_dim
            )
            logging.info(
                "Substituting the last ResNet152 layer with a trainable "
                "Fully Connected Layer to pass from "
                f"{in_features} to {feature_dim} features."
            )

    def freeze_encoder(self) -> None:
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.frozen = True

    def unfreeze_encoder(self) -> None:
        if self.frozen:
            logging.info(f"Encoder model {self.__class__.__name__} started fine-tuning")
            for param in self.resnet.parameters():
                param.requires_grad = True
            self.frozen = False

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.resnet(img)
