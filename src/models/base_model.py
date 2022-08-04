import random
import tempfile
from pathlib import PurePosixPath
from typing import Dict, List

import fasttext
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.data.config import TXT_TRAIN
from src.data.storage import CONFIG_PATH
from src.models.utils.data import FoodPricingDataset, FoodPricingLazyDataset
from src.models.utils.storage import get_local_models_path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class LanguageAndVisionConcat(torch.nn.Module):
    def __init__(
        self,
        loss_fn,
        language_module,
        vision_module,
        language_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
    ):
        super(LanguageAndVisionConcat, self).__init__()
        self.language_module = language_module
        self.vision_module = vision_module
        self.fusion = torch.nn.Linear(
            in_features=(language_feature_dim + vision_feature_dim),
            out_features=fusion_output_size,
        )
        self.fc = torch.nn.Linear(in_features=fusion_output_size, out_features=1)
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, txt, img, label=None):  # TODO: test this None default
        txt_features = torch.nn.functional.relu(self.language_module(txt))
        img_features = torch.nn.functional.relu(self.vision_module(img))
        combined = torch.cat([txt_features, img_features], dim=1)
        fused = self.dropout(torch.nn.functional.relu(self.fusion(combined)))
        pred = self.fc(fused)
        loss = self.loss_fn(pred, label) if label is not None else label
        return (pred, loss)


class FoodPricingBaseModel(LightningModule):
    class DataModule(LightningDataModule):
        def __init__(self, model_instance: LightningModule):
            super().__init__()
            self.hparams = model_instance.hparams
            self.model = model_instance

        def _build_dataset(self, split: str) -> Dataset:
            if self.hparams.lazy_dataset:
                return FoodPricingLazyDataset(
                    img_transform=self.model.img_transform,
                    txt_transform=self.model.txt_transform,
                    split=split,
                )
            else:
                return FoodPricingDataset(
                    img_transform=self.img_transform,
                    txt_transform=self.txt_transform,
                    split=split,
                )

        def train_dataloader(self) -> DataLoader:
            self._train_dataset = self._build_dataset("train")
            return DataLoader(
                dataset=self._train_dataset,
                shuffle=self.hparams.shuffle_train_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.loader_workers,
            )

        def val_dataloader(self) -> DataLoader:
            self._dev_dataset = self._build_dataset("dev")
            return DataLoader(
                dataset=self._dev_dataset,
                shuffle=False,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.loader_workers,
            )

        def test_dataloader(self) -> DataLoader:
            self._test_dataset = self._build_dataset("test")
            return DataLoader(
                dataset=self._test_dataset,
                shuffle=False,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.loader_workers,
            )

    def __init__(self, *args, **kwargs):
        super(FoodPricingBaseModel, self).__init__()
        self.save_hyperparameters()
        self._add_default_hparams()
        self._add_model_specific_hparams()
        self.config: Dict = yaml.safe_load(open(CONFIG_PATH))

        # assign some hparams that get used in multiple places
        self.embedding_dim = self.hparams.embedding_dim
        self.language_feature_dim = self.hparams.language_feature_dim
        self.vision_feature_dim = self.hparams.vision_feature_dim
        self.output_path = self._get_path()

        # build transform models
        self.txt_transform = self._build_txt_transform()
        self.img_transform = self._build_img_transform()

        # Build DataModule
        self.data = self.DataModule(self)

        # set up model and training
        self.model = self._build_model()
        self.trainer_params = self._get_trainer_params()

    # Required LightningModule Methods (when validating)

    def forward(self, txt, img, label=None):
        return self.model(txt, img, label)

    def training_step(self, batch, batch_nb):
        preds, loss = self.forward(
            txt=batch["txt"], img=batch["img"], label=batch["label"]
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_nb):
        preds, loss = self.eval().forward(
            txt=batch["txt"], img=batch["img"], label=batch["label"]
        )
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_epoch_end(self, val_step_outputs):
        avg_loss = torch.stack(tuple(val_step_outputs)).mean()
        self.log("avg_val_loss", avg_loss, logger=True)
        # return {"avg_val_loss": avg_loss, "progress_bar": {"avg_val_loss": avg_loss}}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "avg_val_loss",
                # "frequency": "indicates how often the metric is updated"
                # If "monitor" references validation metrics, then "frequency" should be
                # set to a multiple of "trainer.check_val_every_n_epoch".
            },
        }

    # Convenience Methods

    def fit(self):
        self._set_seed(self.hparams.random_state)
        self.trainer = Trainer(**self.trainer_params)
        # file_path = f"{self.trainer.logger.log_dir}/hparams.yaml"
        # save_hparams_to_yaml(config_yaml=file_path, hparams=self.hparams)
        self.trainer.fit(self)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_txt_transform(self):
        if self.config.get("txt_created", False):
            ft_path = TXT_TRAIN.local_path
            language_transform = fasttext.train_unsupervised(
                str(ft_path),
                model=self.hparams.get("fasttext_model", "cbow"),
                dim=self.embedding_dim,
            )
        else:
            with tempfile.NamedTemporaryFile() as ft_training_data:
                ft_path = PurePosixPath(ft_training_data.name)
                with open(ft_path, "w") as ft:
                    for line in self.train_dataset.iter_txt():
                        ft.write(line + "\n")
                    language_transform = fasttext.train_unsupervised(
                        str(ft_path),
                        model=self.hparams.get("fasttext_model", "cbow"),
                        dim=self.embedding_dim,
                    )
        return language_transform.get_sentence_vector

    def _build_img_transform(self):
        img_dim = self.hparams.img_dim
        img_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(img_dim, img_dim)),
                torchvision.transforms.ToTensor(),
                # all torchvision models expect the same
                # normalization mean and std
                # https://pytorch.org/vision/stable/models.html
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        return img_transform

    def _build_model(self):
        # we're going to pass the outputs of our text
        # transform through an additional trainable layer
        # rather than fine-tuning the transform
        language_module = torch.nn.Linear(
            in_features=self.embedding_dim, out_features=self.language_feature_dim
        )

        # easiest way to get features rather than
        # classification is to overwrite last layer
        # with an identity transformation, we'll reduce
        # dimension using a Linear layer, resnet is 2048 out
        vision_module = torchvision.models.resnet152(pretrained=True)
        for param in vision_module.parameters():
            param.requires_grad = False
        vision_module.fc = torch.nn.Linear(
            in_features=2048, out_features=self.vision_feature_dim
        )

        return LanguageAndVisionConcat(
            loss_fn=torch.nn.MSELoss(),
            language_module=language_module,
            vision_module=vision_module,
            language_feature_dim=self.language_feature_dim,
            vision_feature_dim=self.vision_feature_dim,
            fusion_output_size=self.hparams.fusion_output_size,
            dropout_p=self.hparams.dropout_p,
        )

    def _get_path(
        self, path: List[str] = [], file_name: str = "", file_format: str = ""
    ) -> PurePosixPath:
        return get_local_models_path(path, self, file_name, file_format)

    def _get_trainer_params(self):
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_path,
            filename="{epoch}-{val_loss:.2f}",
            monitor=self.hparams.checkpoint_monitor,
            mode=self.hparams.checkpoint_monitor_mode,
            verbose=self.hparams.verbose,
        )

        early_stop_callback = EarlyStopping(
            monitor=self.hparams.early_stop_monitor,
            min_delta=self.hparams.early_stop_min_delta,
            patience=self.hparams.early_stop_patience,
            verbose=self.hparams.verbose,
        )

        trainer_params = {
            "callbacks": [checkpoint_callback, early_stop_callback],
            "default_root_dir": self.output_path,
            "accumulate_grad_batches": self.hparams.accumulate_grad_batches,
            "accelerator": self.hparams.accelerator,
            "devices": self.hparams.devices,
            "max_epochs": self.hparams.max_epochs,
            "gradient_clip_val": self.hparams.gradient_clip_value,
            "num_sanity_val_steps": self.hparams.num_sanity_val_steps,
        }
        return trainer_params

    @torch.no_grad()
    def make_submission_frame(self) -> pd.DataFrame:
        test_dataloader = self.data.test_dataloader()
        submission_frame = pd.DataFrame(
            index=self.test_dataloader.dataset.index, columns=["true", "pred"]
        )
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            preds, _ = self.model.eval().to("cpu")(batch["txt"], batch["img"])
            submission_frame.loc[batch["id"], "true"] = batch["label"].squeeze(-1)
            submission_frame.loc[batch["id"], "pred"] = preds.squeeze(-1)
        return submission_frame

    def _add_default_hparams(self):
        default_params = {
            "random_state": 42,
            "lazy_dataset": False,
            "shuffle_train_dataset": True,
            "batch_size": 32,
            "loader_workers": 8,  # TODO: set default n_cpu
            # Image and text params
            "img_dim": 224,
            "embedding_dim": 300,
            "language_feature_dim": 300,
            "vision_feature_dim": self.hparams.get("language_feature_dim", 300),
            "fusion_output_size": 512,
            "dropout_p": 0.1,
            # Trainer params
            "verbose": True,
            "accumulate_grad_batches": 1,
            "accelerator": "auto",
            "devices": 1,
            "max_epochs": 100,
            "gradient_clip_val": 1,
            "num_sanity_val_steps": 2,
            # Callback params
            "checkpoint_monitor": "avg_val_loss",
            "checkpoint_monitor_mode": "min",
            "early_stop_monitor": "avg_val_loss",
            "early_stop_min_delta": 0.001,
            "early_stop_patience": 3,
            # Optimizer params
            "lr": 0.001,
        }
        self.hparams.update({**default_params, **self.hparams})

    def _add_model_specific_hparams(self):
        pass
