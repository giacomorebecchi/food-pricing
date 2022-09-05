import tempfile
from datetime import datetime, timezone
from math import ceil
from pathlib import PurePosixPath
from typing import Callable, List

import fasttext
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from ..data.config import TXT_TRAIN
from .base_model import FoodPricingBaseModel
from .utils.data import FoodPricingDataset
from .utils.storage import get_local_models_path, store_submission_frame


class FPMeanBaselineModel:
    def __init__(self, **hparams) -> None:
        self.hparams = hparams
        self.model_name = self.__class__.__name__

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
        if self.hparams.get("store_submission_frame", True):
            store_submission_frame(
                submission_frame=submission_frame,
                model_name=self.model_name,
                run_id=self.hparams.get("trainer_run_id", None),
            )
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

    def _get_dataloader(self, split) -> DataLoader:
        dataset = self._build_dataset(split)
        return DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 8),
        )


class FPCBOWResNet152ConcatBaselineModel(FoodPricingBaseModel):
    def __init__(self, *args, **kwargs):
        super(FPCBOWResNet152ConcatBaselineModel, self).__init__(*args, **kwargs)

    def _build_txt_transform(self) -> Callable:
        if path := self.hparams.fasttext_model_path:
            if run := self.hparams.get("run_path"):
                path = path.replace("/models/", f"/models/{run}/")
            language_transform = fasttext.load_model(path)
        else:
            if self.config.get("txt_created", False):
                ft_path = TXT_TRAIN.local_path
                language_transform = fasttext.train_unsupervised(
                    str(ft_path),
                    model=self.hparams.fasttext_model,
                    dim=self.hparams.embedding_dim,
                )
            else:
                with tempfile.NamedTemporaryFile() as ft_training_data:
                    ft_path = PurePosixPath(ft_training_data.name)
                    with open(ft_path, "w") as ft:
                        train_dataset = self.data.train_dataloader().dataset
                        for line in train_dataset.iter_txt():
                            ft.write(line + "\n")
                        language_transform = fasttext.train_unsupervised(
                            str(ft_path),
                            model=self.hparams.fasttext_model,
                            dim=self.hparams.embedding_dim,
                        )
            path = self._get_fasttext_path()
            self.hparams.update({"fasttext_model_path": path})
            language_transform.save_model(path)
        return language_transform.get_sentence_vector

    def _get_fasttext_path(self) -> str:
        t = datetime.now(timezone.utc).isoformat()
        return str(self._get_path(path=["fasttext"], file_name=t, file_format=".bin"))

    def _add_model_specific_hparams(self) -> None:
        model_specific_hparams = {
            "fasttext_model": "cbow",
            "fasttext_model_path": None,
        }
        self.hparams.update({**model_specific_hparams, **self.hparams})
