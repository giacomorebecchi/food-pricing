import logging
from typing import Generator, Optional

from torch import Tensor, nn
from torchvision.models import resnet152


class PreTrainedResNet152(nn.Module):
    def __init__(
        self,
        weights: Optional[str] = "DEFAULT",
        feature_dim: Optional[int] = None,
    ):
        super(PreTrainedResNet152, self).__init__()
        resnet = resnet152(weights=weights)
        in_features = resnet.fc.in_features
        self.feature_dim = feature_dim
        if self.feature_dim is not None:
            resnet.fc = nn.Linear(in_features=in_features, out_features=feature_dim)
            logging.info(
                "Substituting the last ResNet152 layer with a trainable "
                "Fully Connected Layer to pass from "
                f"{in_features} to {feature_dim} features."
            )
        self.resnet = nn.Sequential(
            resnet,
            nn.ReLU(),
        )
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for name, param in self.named_parameters():
            if self.feature_dim is None or name not in [
                "resnet.0.fc.weight",
                "resnet.0.fc.bias",
            ]:
                param.requires_grad = False
        self.frozen = True

    def unfreeze_encoder(self) -> None:
        if self.frozen:
            for name, param in self.named_parameters():
                if self.feature_dim is None or name not in [
                    "resnet.0.fc.weight",
                    "resnet.0.fc.bias",
                ]:
                    param.requires_grad = True
            self.frozen = False
            logging.info(f"Encoder model {self.__class__.__name__} started fine-tuning")

    def get_general_params(self) -> Generator:
        if self.feature_dim is None:
            yield from ()
        else:
            for name, param in self.named_parameters():
                if name in ["resnet.0.fc.weight", "resnet.0.fc.bias"]:
                    yield param

    def get_encoder_params(self) -> Generator:
        if self.feature_dim is None:
            yield from self.parameters()
        else:
            for name, param in self.named_parameters():
                if name not in ["resnet.0.fc.weight", "resnet.0.fc.bias"]:
                    yield param

    def forward(self, img: Tensor) -> Tensor:
        return self.resnet(img)
