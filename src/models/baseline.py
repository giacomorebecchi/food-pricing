import logging
import random
import tempfile
import warnings
from pathlib import PurePosixPath
from typing import List, Tuple

import fasttext
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
import tqdm

from .utils.data import FoodPricingDataset
from .utils.storage import get_local_models_path

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)


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

    def forward(self, text, image, label=None):  # TODO: test this None default
        text_features = torch.nn.functional.relu(self.language_module(text))
        image_features = torch.nn.functional.relu(self.vision_module(image))
        combined = torch.cat([text_features, image_features], dim=1)
        fused = self.dropout(torch.nn.functional.relu(self.fusion(combined)))
        pred = self.fc(fused)
        loss = self.loss_fn(pred, label) if label is not None else label
        return (pred, loss)


class FPBaselineConcatModel(pl.LightningModule):
    def __init__(self, hparams):
        super(FPBaselineConcatModel, self).__init__()
        self.hparams.update(hparams)

        # assign some hparams that get used in multiple places
        self.embedding_dim = self.hparams.get("embedding_dim", 300)
        self.language_feature_dim = self.hparams.get("language_feature_dim", 300)
        self.vision_feature_dim = self.hparams.get(
            # balance language and vision features by default
            "vision_feature_dim",
            self.language_feature_dim,
        )
        self.output_path = self._get_path(path=["model-outputs"])

        # instantiate dataset
        self._build_dataset()

        # set up model and training
        self.model = self._build_model()
        self.trainer_params = self._get_trainer_params()

    ## Required LightningModule Methods (when validating) ##

    def forward(self, text, image, label=None):
        return self.model(text, image, label)

    def training_step(self, batch, batch_nb):
        preds, loss = self.forward(
            text=batch["text"], image=batch["image"], label=batch["label"]
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        preds, loss = self.eval().forward(
            text=batch["text"], image=batch["image"], label=batch["label"]
        )

        return {"batch_val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(
            tuple(output["batch_val_loss"] for output in outputs)
        ).mean()

        return {"val_loss": avg_loss, "progress_bar": {"avg_val_loss": avg_loss}}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.hparams.get("lr", 0.001)
        )
        return optimizer

        # TODO: Fix the scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "metric_to_track",
        #         "frequency": "indicates how often the metric is updated"
        #         # If "monitor" references validation metrics, then "frequency" should be set to a
        #         # multiple of "trainer.check_val_every_n_epoch".
        #     },
        # }

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset,
            shuffle=False,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16),
        )

    ## Convenience Methods ##

    def fit(self):
        self._set_seed(self.hparams.get("random_state", 42))
        self.trainer = pl.Trainer(**self.trainer_params)
        self.trainer.fit(self)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_text_transform(self):
        with tempfile.NamedTemporaryFile() as ft_training_data:
            ft_path = PurePosixPath(ft_training_data.name)
            with open(ft_path, "w") as ft:
                for line in self.dataset.iter_txt(self.train_dataset.indices):
                    ft.write(line + "\n")
                language_transform = fasttext.train_unsupervised(
                    str(ft_path),
                    model=self.hparams.get("fasttext_model", "cbow"),
                    dim=self.embedding_dim,
                )
        return language_transform.get_sentence_vector

    def _build_image_transform(self):
        image_dim = self.hparams.get("image_dim", 224)
        image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(image_dim, image_dim)),
                torchvision.transforms.ToTensor(),
                # all torchvision models expect the same
                # normalization mean and std
                # https://pytorch.org/vision/stable/models.html
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        return image_transform

    def _build_dataset(self) -> None:
        self.text_transform = lambda x: x
        self.image_transform = self._build_image_transform()
        self.dataset = FoodPricingDataset(
            img_transform=self.image_transform,
            txt_transform=self.text_transform,
        )
        train_len, dev_len, test_len = self._calculate_dataset_ratio()
        (
            self.train_dataset,
            self.dev_dataset,
            self.test_dataset,
        ) = torch.utils.data.random_split(
            self.dataset,
            [train_len, dev_len, test_len],
            generator=torch.Generator().manual_seed(
                (self.hparams.get("random_state", 42))
            ),
        )
        self.txt_transform = self._build_text_transform()
        self.dataset.set_txt_transform(self.txt_transform)

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
        vision_module.fc = torch.nn.Linear(in_features=2048, out_features=1)

        return LanguageAndVisionConcat(
            loss_fn=torch.nn.MSELoss(),
            language_module=language_module,
            vision_module=vision_module,
            language_feature_dim=self.language_feature_dim,
            vision_feature_dim=self.vision_feature_dim,
            fusion_output_size=self.hparams.get("fusion_output_size", 512),
            dropout_p=self.hparams.get("dropout_p", 0.1),
        )

    def _get_path(
        self, path: List[str], file_name: str = "", file_format: str = ""
    ) -> PurePosixPath:
        return get_local_models_path(path, self, file_name, file_format)

    def _get_trainer_params(self):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.output_path,
            filename="{epoch}-{val_loss:.2f}",
            monitor=self.hparams.get("checkpoint_monitor", "avg_val_loss"),
            mode=self.hparams.get("checkpoint_monitor_mode", "min"),
            verbose=self.hparams.get("verbose", True),
        )

        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor=self.hparams.get("early_stop_monitor", "avg_val_loss"),
            min_delta=self.hparams.get("early_stop_min_delta", 0.001),
            patience=self.hparams.get("early_stop_patience", 3),
            verbose=self.hparams.get("verbose", True),
        )

        trainer_params = {
            "callbacks": [checkpoint_callback, early_stop_callback],
            "default_root_dir": self.output_path,
            "accumulate_grad_batches": self.hparams.get("accumulate_grad_batches", 1),
            "gpus": self.hparams.get("n_gpu", 1),
            "max_epochs": self.hparams.get("max_epochs", 100),
            "gradient_clip_val": self.hparams.get("gradient_clip_value", 1),
        }
        return trainer_params

    def _calculate_dataset_ratio(self) -> Tuple[int]:
        try:
            train_ratio, dev_ratio, test_ratio = self.hparams.get("train_dev_test")
            assert train_ratio + dev_ratio + test_ratio == 1.0
        except AssertionError:
            raise Exception(
                "Invalid hyperparameter train_dev_test. Ensure the sum of the three numbers is 1.0"
            )
        except ValueError:
            raise Exception("Expected Tuple of three numbers")
        dev_len = int((len_dataset := len(self.dataset)) * dev_ratio)
        test_len = int(len_dataset * test_ratio)
        train_len = len_dataset - dev_len - test_len
        return train_len, dev_len, test_len

    @torch.no_grad()
    def make_submission_frame(self, test_path):
        test_dataset = self._build_dataset(test_path)
        submission_frame = pd.DataFrame(
            index=test_dataset.samples_frame.id, columns=["proba", "label"]
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16),
        )
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            preds, _ = self.model.eval().to("cpu")(batch["text"], batch["image"])
            submission_frame.loc[batch["id"], "proba"] = preds[:, 1]
            submission_frame.loc[batch["id"], "label"] = preds.argmax(dim=1)
        submission_frame.proba = submission_frame.proba.astype(float)
        submission_frame.label = submission_frame.label.astype(int)
        return submission_frame
