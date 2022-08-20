import random
from typing import TYPE_CHECKING, Callable, Dict, List, Union

import numpy as np
import pandas as pd
import torch
import yaml
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from ..data.storage import CONFIG_PATH
from .utils.callbacks import TelegramBotCallback
from .utils.data import FoodPricingDataset, FoodPricingLazyDataset
from .utils.storage import get_best_checkpoint_path, get_local_models_path

if TYPE_CHECKING:
    from .dual_encoding.pretrained_clip import PreTrainedCLIP
    from .nlp.pretrained_bert import PreTrainedBERT
    from .vision.pretrained_resnet import PreTrainedResNet152


class FoodPricingBaseModel(LightningModule):
    class DataModule(LightningDataModule):
        def __init__(self, model_instance: LightningModule) -> None:
            super().__init__()
            self.hparams.update(model_instance.hparams)
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
                    img_transform=self.model.img_transform,
                    txt_transform=self.model.txt_transform,
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

    def __init__(self, *args, **kwargs) -> None:
        super(FoodPricingBaseModel, self).__init__()
        self.save_hyperparameters()
        self._add_model_specific_hparams()
        self._add_default_hparams()
        self.config: Dict = yaml.safe_load(open(CONFIG_PATH))

        self._set_seed(self.hparams.random_state)

        # build dual model, which has the precedence over other transformers
        if self.hparams.dual_model:
            self.dual_transform = self._build_dual_transform()

        # build transform models
        self.txt_transform = self._build_txt_transform()
        self.img_transform = self._build_img_transform()

        # Build DataModule
        self.data = self.DataModule(self)

        # set up model and training
        self.model = self._build_model()
        self.trainer_params = self._get_trainer_params()

    def forward(self, txt, img, label=None):
        return self.model(txt, img, label)

    def training_step(self, batch: Dict, batch_nb) -> torch.Tensor:
        preds, loss = self.forward(
            txt=batch["txt"], img=batch["img"], label=batch["label"]
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )

        return loss

    def validation_step(self, batch, batch_nb) -> torch.Tensor:
        preds, loss = self.eval().forward(
            txt=batch["txt"], img=batch["img"], label=batch["label"]
        )
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )
        return loss

    def validation_epoch_end(self, val_step_outputs) -> None:
        self.avg_val_loss = torch.stack(
            tuple(val_step_outputs)
        ).mean()  # stored in order to be accessed by Callbacks
        self.log("avg_val_loss", self.avg_val_loss, logger=True)

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.RAdam(
            self.parameters(),
            lr=self.hparams.optimizer_lr,
            weight_decay=self.hparams.optimizer_weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.hparams.lr_scheduler_factor,
            patience=self.hparams.lr_scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "avg_val_loss",
            },
        }

    def fit(self) -> None:
        self._set_seed(self.hparams.random_state)
        self.trainer = Trainer(**self.trainer_params)
        self.trainer.fit(self, datamodule=self.data)

    @classmethod
    def load_from_best_checkpoint(cls, **kwargs) -> LightningModule:
        best_checkpoint_path = get_best_checkpoint_path(model_class=cls, **kwargs)
        return cls.load_from_checkpoint(checkpoint_path=best_checkpoint_path)

    def _set_seed(self, seed) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _is_unfreeze_time(self, module_name: str) -> bool:
        param_name = "n_epochs_unfreeze_" + module_name
        if param_name in self.hparams:
            if self.current_epoch + 1 >= self.hparams[param_name]:
                return True
        return False

    def _unfreeze_module(
        self, module: Union["PreTrainedCLIP", "PreTrainedBERT", "PreTrainedResNet152"]
    ) -> None:
        module.unfreeze_encoder()
        # TODO: Add module.parameters to optimizer

    def _build_dual_transform(self) -> Callable:
        return lambda _: _

    def _build_txt_transform(self) -> Callable:
        return lambda _: _

    def _build_img_transform(self) -> Callable:
        img_dim = self.hparams.img_dim
        img_transform = Compose(
            [
                Resize(size=(img_dim, img_dim)),
                ToTensor(),
                Normalize(mean=self.hparams.img_mean, std=self.hparams.img_std),
            ]
        )
        return img_transform

    def _build_model(self) -> torch.nn.Module:
        pass

    def _get_path(
        self, path: List[str] = [], file_name: str = "", file_format: str = ""
    ) -> str:
        return str(get_local_models_path(path, self, file_name, file_format))

    def _get_trainer_params(self) -> Dict:
        backup_callback = ModelCheckpoint(
            dirpath=self.hparams.output_path,
            filename="backup-{epoch}-{avg_val_loss:.2f}",
            every_n_epochs=self.hparams.backup_n_epochs,
            save_on_train_epoch_end=True,
            verbose=self.hparams.verbose,
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.hparams.output_path,
            filename="{epoch}-{avg_val_loss:.2f}",
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

        notifier_callback = TelegramBotCallback()

        callbacks = [
            backup_callback,
            checkpoint_callback,
            early_stop_callback,
            notifier_callback,
        ]

        trainer_params = {
            "callbacks": callbacks,
            "default_root_dir": self.hparams.output_path,
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
            index=test_dataloader.dataset.index, columns=["true", "pred"]
        )
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            preds, _ = self.model.eval().to("cpu")(batch["txt"], batch["img"])
            submission_frame.loc[batch["id"], "true"] = batch["label"].squeeze(-1)
            submission_frame.loc[batch["id"], "pred"] = preds.squeeze(-1)
        return submission_frame

    def _add_default_hparams(self) -> None:
        default_params = {
            "random_state": 42,
            "lazy_dataset": False,
            "shuffle_train_dataset": True,
            "batch_size": 32,
            "loader_workers": 8,  # TODO: set default n_cpu
            "output_path": self._get_path(),
            # Image and text params
            "img_dim": 224,
            # all torchvision models expect the same
            # normalization mean and std
            # https://pytorch.org/vision/stable/models.html
            "img_mean": [0.485, 0.456, 0.406],
            "img_std": [0.229, 0.224, 0.225],
            "embedding_dim": 300,
            "language_feature_dim": 300,
            "vision_feature_dim": self.hparams.get("language_feature_dim", 300),
            "fusion_output_size": 512,
            "dropout_p": 0.1,
            # Dual model
            "dual_model": False,
            # Trainer params
            "verbose": True,
            "accumulate_grad_batches": 1,
            "accelerator": "auto",
            "devices": 1,
            "max_epochs": 100,
            "gradient_clip_value": 5,
            "num_sanity_val_steps": 2,
            # Callback params
            "checkpoint_monitor": "avg_val_loss",
            "checkpoint_monitor_mode": "min",
            "early_stop_monitor": "avg_val_loss",
            "early_stop_min_delta": 0,
            "early_stop_patience": 10,
            "backup_n_epochs": 10,
            # Optimizer params
            "optimizer_lr": 1e-04,
            "optimizer_weight_decay": 1e-3,
            "lr_scheduler_factor": 0.2,
            "lr_scheduler_patience": 5,
        }
        self.hparams.update({**default_params, **self.hparams})

    def _add_model_specific_hparams(self) -> None:
        pass
