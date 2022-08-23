import itertools
import logging
import random
import traceback
from typing import TYPE_CHECKING, Callable, Dict, Generator, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm.autonotebook import tqdm

from ..data.storage import CONFIG_PATH
from .feature_combinators import (
    LanguageAndVisionConcat,
    LanguageAndVisionWeightedImportance,
)
from .utils.callbacks import TelegramBotCallback
from .utils.data import FoodPricingDataset, FoodPricingLazyDataset
from .utils.storage import (
    get_best_checkpoint_path,
    get_local_models_path,
    store_submission_frame,
)
from .vision.pretrained_resnet import PreTrainedResNet152

if TYPE_CHECKING:
    from .dual_encoding.pretrained_clip import PreTrainedCLIP
    from .nlp.pretrained_bert import PreTrainedBERT


class FoodPricingBaseModel(LightningModule):
    class DataModule(LightningDataModule):
        def __init__(self, model_instance: LightningModule) -> None:
            super().__init__()
            self.hparams.update(model_instance.hparams)
            self.model = model_instance
            self.generator = torch.Generator()
            self.generator.manual_seed(self.model.hparams.random_seed)

            # initialize datasets
            self.train_dataset = self._build_dataset("train")
            self.dev_dataset = self._build_dataset("dev")
            self.test_dataset = self._build_dataset("test")

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
            return DataLoader(
                dataset=self.train_dataset,
                shuffle=self.hparams.shuffle_train_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.loader_workers,
                worker_init_fn=self._seed_worker,
                generator=self.generator,
            )

        def val_dataloader(self) -> DataLoader:
            return DataLoader(
                dataset=self.dev_dataset,
                shuffle=False,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.loader_workers,
                worker_init_fn=self._seed_worker,
                generator=self.generator,
            )

        def test_dataloader(self) -> DataLoader:
            return DataLoader(
                dataset=self.test_dataset,
                shuffle=False,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.loader_workers,
                worker_init_fn=self._seed_worker,
                generator=self.generator,
            )

        def _seed_worker(self, worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

    def __init__(self, *args, **kwargs) -> None:
        super(FoodPricingBaseModel, self).__init__()
        self.save_hyperparameters()
        self._add_model_specific_hparams()
        self._add_default_hparams()
        self.config: Dict = yaml.safe_load(open(CONFIG_PATH))

        self.model_name = self.__class__.__name__

        self._set_seed(self.hparams.random_seed)

        # build dual module, which has the precedence over other transformers
        if self.hparams.dual_module:
            self.dual_module = self._build_dual_module()
        # else, build the language and vision modules separately
        else:
            self.language_module = self._build_txt_module()
            self.vision_module = self._build_img_module()

        # build transform models
        self.txt_transform = self._build_txt_transform()
        self.img_transform = self._build_img_transform()

        # Build DataModule
        self.data = self.DataModule(self)

        # set up model and training
        if self.hparams.attention_module:
            self.attention_module = self._build_attention_module()
        self.fusion_module = self._build_fusion_module()
        self.trainer_params = self._get_trainer_params()

        # defining the loss function
        self.loss_fn = torch.nn.MSELoss()

    def on_train_epoch_start(self) -> None:
        self.training_step_outputs = []
        if self.hparams.dual_module:
            if self._is_unfreeze_time("dual_module"):
                self._unfreeze_module(self.dual_module)

        else:
            if self._is_unfreeze_time("language_module"):
                self._unfreeze_module(self.language_module)

            if self._is_unfreeze_time("vision_module"):
                self._unfreeze_module(self.vision_module)

    def forward(self, txt, img, label=None):
        if self.hparams.dual_module:
            txt, img = self.dual_module(txt, img)
        else:
            txt = self.language_module(txt)
            img = self.vision_module(img)
        if self.hparams.attention_module:
            txt, img = self.attention_module(txt, img)
        pred = self.fusion_module(txt, img)
        loss = self.loss_fn(pred, label) if label is not None else None
        return pred, loss

    def training_step(self, batch: Dict, batch_nb, optimizer_idx=None) -> torch.Tensor:
        _, loss = self.forward(txt=batch["txt"], img=batch["img"], label=batch["label"])
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

    def training_epoch_end(self, training_step_outputs) -> None:
        logging.info(training_step_outputs)
        self.avg_train_loss = torch.Tensor(
            self._stack_outputs(training_step_outputs)
        ).mean()  # stored in order to be accessed by Callbacks
        self.log("avg_train_loss", self.avg_train_loss, logger=True)

    def validation_step(self, batch, batch_nb) -> torch.Tensor:
        _, loss = self.eval().forward(
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

    def configure_optimizers(self) -> Union[Dict, Tuple[Dict, Dict]]:
        general_params = self._get_general_params()

        if self.hparams.optimizer_name == "radam":
            optimizer = torch.optim.RAdam(
                general_params,
                lr=self.hparams.optimizer_lr,
                weight_decay=self.hparams.optimizer_weight_decay,
            )
        elif self.hparams.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                general_params,
                lr=self.hparams.optimizer_lr,
                weight_decay=self.hparams.optimizer_weight_decay,
            )
        else:
            raise Exception(f"Uninmplemented optimizer {self.hparams.optimizer_name}.")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.hparams.lr_scheduler_factor,
            patience=self.hparams.lr_scheduler_patience,
        )

        optim_config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "avg_val_loss",
            },
        }

        if self.hparams.encoder_optimizer_name is not None:
            # if not None, this means we want different optimizers for the
            # encoder modules and the general parameters
            encoder_params = self._get_encoder_params()
            if self.hparams.encoder_optimizer_name == "radam":
                encoder_optimizer = torch.optim.RAdam(
                    encoder_params,
                    lr=self.hparams.encoder_optimizer_lr,
                    weight_decay=self.hparams.encoder_optimizer_weight_decay,
                )
            elif self.hparams.encoder_optimizer_name == "adamw":
                encoder_optimizer = torch.optim.AdamW(
                    encoder_params,
                    lr=self.hparams.encoder_optimizer_lr,
                    weight_decay=self.hparams.encoder_optimizer_weight_decay,
                )
            else:
                raise Exception(
                    f"Uninmplemented optimizer {self.hparams.optimizer_name}."
                )

            encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                encoder_optimizer,
                factor=self.hparams.encoder_lr_scheduler_factor,
                patience=self.hparams.encoder_lr_scheduler_patience,
            )

            encoder_optim_config = {
                "optimizer": encoder_optimizer,
                "lr_scheduler": {
                    "scheduler": encoder_scheduler,
                    "monitor": "avg_val_loss",
                },
            }

            optim_config = (  # this becomes a Tuple
                optim_config,
                encoder_optim_config,
            )

        return optim_config

    def fit(self) -> None:
        self._set_seed(self.hparams.random_seed)
        self.trainer = Trainer(**self.trainer_params)
        self.trainer.fit(self, datamodule=self.data)

    @classmethod
    def load_from_best_checkpoint(cls, **kwargs) -> LightningModule:
        best_checkpoint_path = get_best_checkpoint_path(model_class=cls, **kwargs)
        return cls.load_from_checkpoint(checkpoint_path=best_checkpoint_path)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _is_unfreeze_time(self, module_name: str) -> bool:
        if module_name == "dual_module":
            if not self.hparams.dual_module:
                return False  # if dual_module is not in the architecture
        param_name = "n_epochs_unfreeze_" + module_name
        return (param_name in self.hparams) and (
            self.current_epoch >= self.hparams[param_name]
        )

    def _unfreeze_module(
        self, module: Union["PreTrainedCLIP", "PreTrainedBERT", "PreTrainedResNet152"]
    ) -> None:
        try:
            module.unfreeze_encoder()
        except Exception:
            trbck = traceback.format_exc()
            message = (
                f"Attempted unfreezing module {module.__class__.__name__}.\n"
                + f"Complete traceback: {trbck}"
            )
            logging.info(message)

    def _get_general_params(self) -> Generator:
        if self.hparams.encoder_optimizer_name is None:
            return self.parameters()
        else:
            params = [self.fusion_module.parameters()]
            if self.hparams.attention_module:
                params.append(self.attention_module.parameters())

            if self.hparams.dual_module:
                encoders = [self.dual_module]
            else:
                encoders = [self.language_module, self.vision_module]
            for encoder in encoders:
                try:
                    params.append(encoder.get_general_params())
                except Exception:
                    logging.info(
                        f"Unsuccessfully loaded general parameters in module: {encoder}"
                    )
        return itertools.chain(*params)

    def _get_encoder_params(self) -> Generator:
        if self.hparams.dual_module:
            encoders = [self.dual_module]
        else:
            encoders = [self.language_module, self.vision_module]
        params = []
        for encoder in encoders:
            try:
                params.append(encoder.get_encoder_params())
            except Exception:
                logging.info(
                    f"Unsuccessfully loaded encoder parameters in module: {encoder}"
                )

        return itertools.chain(*params)

    def _stack_outputs(self, outputs) -> torch.Tensor:
        if isinstance(outputs, list):
            return [self._stack_outputs(output) for output in outputs]
        elif isinstance(outputs, dict):
            return outputs["loss"]

    def _build_dual_module(self) -> torch.nn.Module:
        return lambda txt, img: (txt, img)

    def _build_txt_module(self) -> torch.nn.Module:
        module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.hparams.embedding_dim,
                out_features=self.hparams.language_feature_dim,
            ),
            torch.nn.ReLU(),
        )
        return module

    def _build_img_module(self) -> torch.nn.Module:
        module = PreTrainedResNet152(feature_dim=self.hparams.vision_feature_dim)
        return module

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

    def _build_attention_module(self) -> torch.nn.Module:
        module = LanguageAndVisionWeightedImportance(
            language_feature_dim=self.hparams.language_feature_dim,
            vision_feature_dim=self.hparams.vision_feature_dim,
        )
        return module

    def _build_fusion_module(self) -> torch.nn.Module:
        module = LanguageAndVisionConcat(
            language_feature_dim=self.hparams.language_feature_dim,
            vision_feature_dim=self.hparams.vision_feature_dim,
            fusion_output_size=self.hparams.fusion_output_size,
            dropout_p=self.hparams.dropout_p,
        )
        return module

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
            preds, _ = self.eval().to("cpu")(batch["txt"], batch["img"])
            submission_frame.loc[batch["id"], "true"] = batch["label"].squeeze(-1)
            submission_frame.loc[batch["id"], "pred"] = preds.squeeze(-1)

        if self.hparams.store_submission_frame:
            store_submission_frame(
                submission_frame=submission_frame,
                model_name=self.model_name,
                run_id=self.hparams.trainer_run_id,
            )

        return submission_frame

    def _add_default_hparams(self) -> None:
        default_params = {
            "random_seed": 42,
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
            # Dual module
            "dual_module": False,
            # Attention module
            "attention_module": False,
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
            "optimizer_name": "adamw",
            "optimizer_lr": 1e-04,
            "optimizer_weight_decay": 1e-3,
            "lr_scheduler_factor": 0.2,
            "lr_scheduler_patience": 5,
            # Specific Encoders optimizer params
            "encoder_optimizer_name": None,  # w/ None, encoder uses the standard opt.
            "encoder_optimizer_lr": self.hparams.get("optimizer_lr", 1e-04),
            "encoder_optimizer_weight_decay": self.hparams.get(
                "optimizer_weight_decay", 1e-03
            ),
            # Test evaluation stored
            "store_submission_frame": True,
            "trainer_run_id": None,
            # Modules unfreezing
            "n_epochs_unfreeze_language_module": 10,
            "n_epochs_unfreeze_vision_module": 20,
            "n_epochs_unfreeze_dual_module": 10,
        }
        self.hparams.update({**default_params, **self.hparams})

    def _add_model_specific_hparams(self) -> None:
        pass
