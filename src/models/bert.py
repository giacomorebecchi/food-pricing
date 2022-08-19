import torch
import torchvision

from .base_model import FoodPricingBaseModel
from .feature_combinators import (
    LanguageAndVisionConcat,
    LanguageAndVisionWeightedImportance,
)
from .nlp.pretrained_bert import PreTrainedBERT


class FPBERTResNet152ConcatModel(FoodPricingBaseModel):
    def __init__(self, *args, **kwargs):
        super(FPBERTResNet152ConcatModel, self).__init__(*args, **kwargs)

    def _build_txt_transform(self) -> PreTrainedBERT:
        model_kwargs = {"pretrained_model_name_or_path": self.hparams.bert_model}
        tokenizer_kwargs = {
            "pretrained_model_name_or_path": self.hparams.tokenizer_bert_model
        }
        language_transform = PreTrainedBERT(
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            feature_dim=None,
        )
        self.hparams.update(
            {
                "embedding_dim": language_transform.encoder_features,
            }
        )
        return language_transform

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

        return LanguageAndVisionConcat(
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
            "bert_model": "dbmdz/bert-base-italian-xxl-uncased",
            "tokenizer_bert_model": self.hparams.get(
                "bert_model", "dbmdz/bert-base-italian-xxl-uncased"
            ),  # Default is same as "bert_model" param
        }
        self.hparams.update({**model_specific_hparams, **self.hparams})


class FPBERTResNet152WeightedConcatModel(FPBERTResNet152ConcatModel):
    def __init__(self, *args, **kwargs):
        super(FPBERTResNet152WeightedConcatModel, self).__init__(*args, **kwargs)

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

        self.weighting_module = LanguageAndVisionWeightedImportance(
            language_module=language_module,
            vision_module=vision_module,
            language_feature_dim=self.hparams.language_feature_dim,
            vision_feature_dim=self.hparams.vision_feature_dim,
        )

        return LanguageAndVisionConcat(
            loss_fn=torch.nn.MSELoss(),
            dual_module=self.weighting_module,
            language_feature_dim=self.hparams.language_feature_dim,
            vision_feature_dim=self.hparams.vision_feature_dim,
            fusion_output_size=self.hparams.fusion_output_size,
            dropout_p=self.hparams.dropout_p,
        )
