import logging
from typing import Dict, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)


class PreTrainedBERT(torch.nn.Module):
    def __init__(
        self,
        model_kwargs: Dict,
        tokenizer_kwargs: Optional[Dict] = {},
    ) -> None:
        super(PreTrainedBERT, self).__init__()
        self.pretrained_model_name_or_path = model_kwargs.get(
            "pretrained_model_name_or_path", 0
        )
        if not self.pretrained_model_name_or_path:
            raise ValueError("Specify the parameter pretrained_model_name_or_path.")
        self.bert = AutoModel.from_pretrained(**model_kwargs, output_hidden_states=True)
        self.encoder_features = self.bert.config.dim
        if not tokenizer_kwargs.get("pretrained_model_name_or_path", 0):
            logging.info(
                """
                Loading the tokenizer with the same name
                of the model for the PretrainedBERT.
                """
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
            logging.info("Encoder model fine-tuning")
            for param in self.bert.parameters():
                param.requires_grad = True
            self.frozen = False

    def prepare_sample(self, text_sample: List[str]) -> Dict:
        return self.tokenizer(text_sample, padding=True, return_tensors="pt")

    def forward(self, txt):
        encoded_batch = self.prepare_sample(txt)
        token_emb = self.bert(
            encoded_batch["input_ids"], encoded_batch["attention_mask"]
        )
        return token_emb[0][:, 0, :]
