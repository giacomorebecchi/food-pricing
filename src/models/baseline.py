import tempfile
from datetime import datetime, timezone
from math import ceil
from pathlib import PurePosixPath
from typing import List

import fasttext
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.config import TXT_TRAIN
from .base_model import FoodPricingBaseModel
from .feature_combinators import LanguageAndVisionConcat
from .utils.data import FoodPricingDataset
from .utils.storage import get_local_models_path
from .vision.pretrained_resnet import PreTrainedResNet152


class FPMeanBaselineModel:
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


class FPCBOWResNet152ConcatBaselineModel(FoodPricingBaseModel):
    def __init__(self, *args, **kwargs):
        super(FPCBOWResNet152ConcatBaselineModel, self).__init__(*args, **kwargs)

    def on_epoch_end(self) -> None:
        if self._is_unfreeze_time("vision_module"):
            self._unfreeze_module(self.vision_module)

    def _build_txt_transform(self):
        if path := self.hparams.fasttext_model_path:
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

    def _get_fasttext_path(self):
        t = datetime.now(timezone.utc).isoformat()
        return str(self._get_path(path=["fasttext"], file_name=t, file_format=".bin"))

    def _build_model(self):
        self.language_module = torch.nn.Linear(
            in_features=self.hparams.embedding_dim,
            out_features=self.hparams.language_feature_dim,
        )

        self.vision_module = PreTrainedResNet152(
            feature_dim=self.hparams.vision_feature_dim
        )

        return LanguageAndVisionConcat(
            loss_fn=torch.nn.MSELoss(),
            language_module=self.language_module,
            vision_module=self.vision_module,
            language_feature_dim=self.hparams.language_feature_dim,
            vision_feature_dim=self.hparams.vision_feature_dim,
            fusion_output_size=self.hparams.fusion_output_size,
            dropout_p=self.hparams.dropout_p,
        )

    def _add_model_specific_hparams(self) -> None:
        model_specific_hparams = {
            "dropout_p": 0.2,
            "fusion_output_size": 512,
            "fasttext_model": "cbow",
            "fasttext_model_path": None,
            "n_epochs_unfreeze_vision_module": 10,
        }
        self.hparams.update({**model_specific_hparams, **self.hparams})
