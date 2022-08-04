from math import ceil
from pathlib import PurePosixPath
from typing import List

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils.data import FoodPricingDataset
from .utils.storage import get_local_models_path


class FPBaselineMeanModel:
    def __init__(self, **hparams) -> None:
        self.hparams = hparams
        self.output_path = self._get_path()

    def fit(self) -> None:
        train_dataloader = self._get_dataloader("train")
        n = len(train_dataloader.dataset)
        ar = np.zeros(n)
        for i, batch in tqdm(
            enumerate(train_dataloader),
            total=ceil(n / (size := train_dataloader.batch_size)),
        ):
            ar[size * i : min(n, size * (i + 1))] = batch["label"].squeeze(-1)
        self.pred = ar.mean()
        print(f"Finished training with MSE: {ar.std()**2:.3f}")

    def make_submission_frame(self) -> pd.DataFrame:
        test_dataloader = self._get_dataloader("test")
        submission_frame = pd.DataFrame(
            index=test_dataloader.dataset.index, columns=["true", "pred"]
        )
        n_batches = ceil(len(test_dataloader.dataset) / test_dataloader.batch_size)
        for batch in tqdm(test_dataloader, total=n_batches):
            submission_frame.loc[batch["id"], "true"] = batch["label"].squeeze(-1)
            submission_frame.loc[batch["id"], "pred"] = self.pred
        test_mse = ((submission_frame["true"] - submission_frame["pred"]).pow(2)).mean()
        print(f"Test MSE: {test_mse:.3f}")
        return submission_frame

    def _build_dataset(self, split: str) -> FoodPricingDataset:
        return FoodPricingDataset(
            img_transform=lambda _: np.nan,
            txt_transform=lambda _: np.nan,
            split=split,
        )

    def _get_path(
        self, path: List[str] = [], file_name: str = "", file_format: str = ""
    ) -> PurePosixPath:
        return get_local_models_path(path, self, file_name, file_format)

    def _get_dataloader(self, split):
        dataset = self._build_dataset(split)
        return DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 8),
        )
