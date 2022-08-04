import tempfile
from pathlib import PurePosixPath

import fasttext
import torch
import torchvision

from ..data.config import TXT_TRAIN
from .base_model import FoodPricingBaseModel


class FoodPricingConcatModel(FoodPricingBaseModel):
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
            super().__init__()
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

    def __init__(self, *args, **kwargs):
        super(FoodPricingConcatModel, self).__init__(*args, **kwargs)

    def _build_txt_transform(self):
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
        return language_transform.get_sentence_vector

    def _build_model(self):
        # we're going to pass the outputs of our text
        # transform through an additional trainable layer
        # rather than fine-tuning the transform
        language_module = torch.nn.Linear(
            in_features=self.hparams.embedding_dim,
            out_features=self.hparams.language_feature_dim,
        )

        # easiest way to get features rather than
        # classification is to overwrite last layer
        # with an identity transformation, we'll reduce
        # dimension using a Linear layer, resnet is 2048 out
        vision_module = torchvision.models.resnet152(weights="DEFAULT")
        for param in vision_module.parameters():
            param.requires_grad = False
        vision_module.fc = torch.nn.Linear(
            in_features=2048, out_features=self.hparams.vision_feature_dim
        )

        return self.LanguageAndVisionConcat(
            loss_fn=torch.nn.MSELoss(),
            language_module=language_module,
            vision_module=vision_module,
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
        }
        self.hparams.update({**model_specific_hparams, **self.hparams})
