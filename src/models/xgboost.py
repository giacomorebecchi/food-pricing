import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xgboost import DMatrix

from .utils.data import FoodPricingDataset


class XGBBaseModel:
    def __init__(self, **kwargs):
        self.save_hyperparameters(kwargs)
        self.txt_transform = self._build_txt_transform()
        self.img_transform = self._build_img_transform()
        self.train_dataset = self._build_dataset("train")
        self.dev_dataset = self._build_dataset("dev")
        self.test_dataset = self._build_dataset("test")

    def save_hyperparameters(self, kwargs):
        self.hparams = self._load_default_hparams()
        self.hparams.update(kwargs)

    def _load_default_hparams(self):
        default_hparams = {}
        return default_hparams

    def _build_txt_transform(self):
        pass

    def _build_img_transform(self):
        pass

    def _build_dataset(self, split: str) -> DMatrix:
        dataset = FoodPricingDataset(
            img_transform=self.img_transform,
            txt_transform=self.txt_transform,
            split=split,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=self.hparams["shuffle"],
            num_workers=self.hparams["loader_workers"],
        )
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
        labels = ar[:, 0]
        features = ar[:, 1:]
        return DMatrix(data=features, label=labels)
