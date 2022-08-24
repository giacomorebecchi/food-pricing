from .base_model import FoodPricingBaseModel
from .nlp.pretrained_bert import PreTrainedBERT


class FPBERTResNet152ConcatModel(FoodPricingBaseModel):
    def __init__(self, *args, **kwargs):
        super(FPBERTResNet152ConcatModel, self).__init__(*args, **kwargs)

    def _build_txt_module(self) -> PreTrainedBERT:
        model_kwargs = {"pretrained_model_name_or_path": self.hparams.bert_model}
        tokenizer_kwargs = {
            "pretrained_model_name_or_path": self.hparams.tokenizer_bert_model
        }
        language_module = PreTrainedBERT(
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            feature_dim=self.hparams.language_feature_dim,
        )
        self.hparams.update(
            {
                "embedding_dim": language_module.encoder_features,
            }
        )
        return language_module

    def _add_model_specific_hparams(self) -> None:
        model_specific_hparams = {
            "bert_model": "dbmdz/bert-base-italian-xxl-uncased",
            "tokenizer_bert_model": self.hparams.get(
                "bert_model", "dbmdz/bert-base-italian-xxl-uncased"
            ),  # Default is same as "bert_model" param
        }
        self.hparams.update({**model_specific_hparams, **self.hparams})


class FPBERTResNet152WeightedConcatModel(FPBERTResNet152ConcatModel):
    def __init__(self, *args, **kwargs):
        super(FPBERTResNet152WeightedConcatModel, self).__init__(*args, **kwargs)

    def _add_model_specific_hparams(self) -> None:
        model_specific_hparams = {
            "attention_module": True,
            "bert_model": "dbmdz/bert-base-italian-xxl-uncased",
            "tokenizer_bert_model": self.hparams.get(
                "bert_model", "dbmdz/bert-base-italian-xxl-uncased"
            ),  # Default is same as "bert_model" param
        }
        self.hparams.update({**model_specific_hparams, **self.hparams})
