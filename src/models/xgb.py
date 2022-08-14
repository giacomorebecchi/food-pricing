from typing import Any, Dict, Tuple

import numpy as np
import torch
from pytorch_lightning.utilities.parsing import AttributeDict
from torch.utils.data import DataLoader
from torchvision.models import resnet152
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from .nlp.pretrained_bert import PreTrainedBERT
from .utils.data import FoodPricingDataset
from .utils.storage import get_local_models_path


class XGBBaseModel:
    def __init__(self, **kwargs):
        self.save_hyperparameters(kwargs)
        self.txt_transform = self._build_txt_transform()
        self.img_transform = self._build_img_transform()
        self.y_train, self.X_train = self._build_dataset("train")
        self.y_dev, self.X_dev = self._build_dataset("dev")
        self.y_test, self.X_test = self._build_dataset("test")

    def save_hyperparameters(self, kwargs: Dict[str, Any]) -> None:
        self.hparams = AttributeDict(kwargs)
        self._add_model_specific_hparams()
        self._add_default_hparams()

    def _add_default_hparams(self) -> None:
        default_params = {
            "batch_size": 32,
            "loader_workers": 8,
            "shuffle": False,
            "img_dim": 224,
            # all torchvision models expect the same
            # normalization mean and std
            # https://pytorch.org/vision/stable/models.html
            "img_mean": [0.485, 0.456, 0.406],
            "img_std": [0.229, 0.224, 0.225],
        }
        self.hparams.update({**default_params, **self.hparams})

    def _add_model_specific_hparams(self) -> None:
        pass

    def _build_txt_transform(self):
        pass

    def _build_img_transform(self):
        pass

    def _build_dataset(self, split: str) -> Tuple[np.ndarray]:
        dataset = FoodPricingDataset(
            img_transform=self.img_transform,
            txt_transform=self.txt_transform,
            split=split,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle and split == "train",
            num_workers=self.hparams.loader_workers,
        )
        y_X = self._get_data(dataloader, split)
        path = self._get_data_path(split)
        np.save(path, y_X)
        return y_X[:, 0], y_X[:, 1:]  # y, X

    def _get_data_path(self, split: str) -> str:
        return get_local_models_path(
            path=["data"], model=self, file_name=split, file_format=".npy"
        )

    def _get_data(self, dataloader: DataLoader, split: str) -> np.ndarray:
        ar = torch.vstack(
            [
                torch.hstack(
                    (
                        batch["label"],
                        torch.squeeze(batch["img"]),
                        torch.squeeze(batch["txt"]),
                    )
                )
                for batch in tqdm(dataloader, desc=f"Creating {split} dataset: ")
            ]
        ).numpy()
        return ar


class XGBBERTResNet152(XGBBaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_txt_transform(self):
        model_kwargs = {"pretrained_model_name_or_path": self.hparams.bert_model}
        tokenizer_kwargs = {
            "pretrained_model_name_or_path": self.hparams.tokenizer_bert_model
        }
        language_transform = PreTrainedBERT(
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self.hparams.update(
            {
                "embedding_dim": language_transform.encoder_features,
            }
        )
        return language_transform

    def _build_img_transform(self):
        module = resnet152(weights="DEFAULT")
        img_dim = self.hparams.img_dim
        transformer = Compose(
            [
                Resize(size=(img_dim, img_dim)),
                ToTensor(),
                Normalize(mean=self.hparams.img_mean, std=self.hparams.img_std),
            ]
        )
        for param in module.parameters():
            param.requires_grad = False
        return lambda img: module(torch.unsqueeze(transformer(img), 0))

    def _add_model_specific_hparams(self) -> None:
        model_specific_hparams = {
            "bert_model": "dbmdz/bert-base-italian-xxl-uncased",
            "tokenizer_bert_model": self.hparams.get(
                "bert_model", "dbmdz/bert-base-italian-xxl-uncased"
            ),
        }
        self.hparams.update({**model_specific_hparams, **self.hparams})
