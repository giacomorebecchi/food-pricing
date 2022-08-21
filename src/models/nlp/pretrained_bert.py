import logging
from typing import Dict, List, Optional

from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)


class PreTrainedBERT(nn.Module):
    def __init__(
        self,
        model_kwargs: Dict,
        tokenizer_kwargs: Optional[Dict] = {},
        feature_dim: Optional[int] = None,
    ) -> None:
        super(PreTrainedBERT, self).__init__()
        self.pretrained_model_name_or_path = model_kwargs.get(
            "pretrained_model_name_or_path", 0
        )
        if not self.pretrained_model_name_or_path:
            raise ValueError("Specify the parameter pretrained_model_name_or_path.")
        self.bert = AutoModel.from_pretrained(**model_kwargs, output_hidden_states=True)
        self.encoder_features = self.bert.config.hidden_size
        self.add_fc = False
        if feature_dim and feature_dim != self.encoder_features:
            self.add_fc = True
            self.fc = nn.Sequential(
                nn.Linear(in_features=self.encoder_features, out_features=feature_dim),
                nn.ReLU(),
            )
            logging.info(
                "Adding a Fully Connected Layer to pass from "
                f"{self.encoder_features} to {feature_dim} features."
            )
        if not tokenizer_kwargs.get("pretrained_model_name_or_path", 0):
            logging.info(
                "Loading the tokenizer with the same name "
                "of the model for the PretrainedBERT."
            )
            tokenizer_kwargs[
                "pretrained_model_name_or_path"
            ] = self.pretrained_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for param in self.bert.parameters():
            param.requires_grad = False
        self.frozen = True

    def unfreeze_encoder(self) -> None:
        if self.frozen:
            logging.info(f"Encoder model {self.__class__.__name__} started fine-tuning")
            for param in self.bert.parameters():
                param.requires_grad = True
            self.frozen = False

    def prepare_sample(self, txt: List[str]) -> Dict:
        return self.tokenizer(txt, padding=True, return_tensors="pt")

    def forward(self, txt: List[str]) -> Tensor:
        encoded_batch = self.prepare_sample(txt)
        token_emb = self.bert(
            encoded_batch["input_ids"], encoded_batch["attention_mask"]
        )
        sent_emb = token_emb[0][:, 0, :]  # [CLS Token]
        if self.add_fc:
            return self.fc(sent_emb)
        else:
            return sent_emb
