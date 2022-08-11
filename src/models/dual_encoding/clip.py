import logging
from typing import Dict, Optional

import torch
from transformers import AutoProcessor, VisionTextDualEncoderModel


class PreTrainedCLIP(torch.nn.Module):
    def __init__(
        self,
        model_kwargs: Dict,
        processor_kwargs: Optional[Dict] = {},
        img_feature_dim: Optional[int] = None,
        txt_feature_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.pretrained_model_name_or_path = model_kwargs.get(
            "pretrained_model_name_or_path", 0
        )
        if not self.pretrained_model_name_or_path:
            raise ValueError("Specify the parameter pretrained_model_name_or_path.")
        self.clip = VisionTextDualEncoderModel.from_pretrained(**model_kwargs)
        if not processor_kwargs.get("pretrained_model_name_or_path", 0):
            logging.info(
                "Loading the tokenizer with the same name "
                "of the model for the PretrainedBERT."
            )
            processor_kwargs[
                "pretrained_model_name_or_path"
            ] = self.pretrained_model_name_or_path
        processor_kwargs_default = {
            "do_resize": False,
            "do_convert_rgb": False,
            "do_center_crop": False,
            "do_normalize": False,
        }
        processor_kwargs_default.update(processor_kwargs)
        self.processor = AutoProcessor.from_pretrained(**processor_kwargs_default)
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for param in self.clip.parameters():
            param.requires_grad = False
        self.frozen = True

    def unfreeze_encoder(self) -> None:
        if self.frozen:
            logging.info("Encoder model fine-tuning")
            for param in self.clip.parameters():
                param.requires_grad = True
            self.frozen = False

    def forward(self, sample):
        pass
