import logging
from typing import Dict, Generator, List, Optional, Tuple

from torch import Tensor, nn, tensor
from transformers import AutoProcessor, VisionTextDualEncoderModel


class PreTrainedCLIP(nn.Module):
    def __init__(
        self,
        model_kwargs: Dict,
        processor_kwargs: Optional[Dict] = {},
        img_feature_dim: Optional[int] = None,
        txt_feature_dim: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.return_tensors = return_tensors
        self.pretrained_model_name_or_path = model_kwargs.get(
            "pretrained_model_name_or_path", 0
        )
        if not self.pretrained_model_name_or_path:
            raise ValueError("Specify the parameter pretrained_model_name_or_path.")
        self.clip = VisionTextDualEncoderModel.from_pretrained(**model_kwargs)
        self.encoder_features = self.clip.projection_dim
        self.add_img_fc, self.add_txt_fc = False, False
        if img_feature_dim and img_feature_dim != self.encoder_features:
            self.add_img_fc = True
            self.img_fc = nn.Sequential(
                nn.Linear(
                    in_features=self.encoder_features, out_features=img_feature_dim
                ),
                nn.ReLU(),
            )
            logging.info(
                "Adding a Fully Connected Layer to image embeddings to pass from "
                f"{self.encoder_features} to {img_feature_dim} features."
            )
        if txt_feature_dim and txt_feature_dim != self.encoder_features:
            self.add_txt_fc = True
            self.txt_fc = nn.Sequential(
                nn.Linear(
                    in_features=self.encoder_features, out_features=txt_feature_dim
                ),
                nn.ReLU(),
            )
            logging.info(
                "Adding a Fully Connected Layer to text embeddings to pass from "
                f"{self.encoder_features} to {txt_feature_dim} features."
            )
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
        self.img_size = self.processor.feature_extractor.crop_size
        self.img_mean = self.processor.feature_extractor.image_mean
        self.img_std = self.processor.feature_extractor.image_std
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for param in self.clip.parameters():
            param.requires_grad = False
        self.frozen = True

    def unfreeze_encoder(self) -> None:
        if self.frozen:
            logging.info(f"Encoder model {self.__class__.__name__} started fine-tuning")
            for param in self.clip.parameters():
                param.requires_grad = True
            self.frozen = False

    def prepare_sample(self, txt: List[str], img: Tensor) -> Dict[str, Tensor]:
        inputs = self.processor(
            text=txt,
            images=img,
            return_tensors=self.return_tensors,
            padding=True,
        )
        if self.return_tensors is None:
            inputs["input_ids"] = tensor(inputs["input_ids"])
            inputs["attention_mask"] = tensor(inputs["attention_mask"])
            if (
                isinstance(inputs["pixel_values"], list)
                and len(inputs["pixel_values"]) == 1
            ):
                inputs["pixel_values"] = inputs["pixel_values"][0]
            else:
                raise ValueError("Pixel values could not be transformed into a tensor.")
        return inputs

    def get_general_params(self) -> Generator:
        if self.add_txt_fc or self.add_img_fc:
            for name, param in self.named_parameters():
                if name in [
                    "img_fc.0.weight",
                    "img_fc.0.bias",
                    "txt_fc.0.weight",
                    "txt_fc.0.bias",
                ]:
                    yield param
        else:
            yield from ()

    def get_encoder_params(self) -> Generator:
        if self.add_txt_fc or self.add_img_fc:
            for name, param in self.named_parameters():
                if name not in [
                    "img_fc.0.weight",
                    "img_fc.0.bias",
                    "txt_fc.0.weight",
                    "txt_fc.0.bias",
                ]:
                    yield param
        else:
            yield from self.parameters()

    def forward(self, txt: List[str], img: Tensor) -> Tuple[Tensor, Tensor]:
        inputs = self.prepare_sample(txt, img)
        outputs = self.clip(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            return_loss=False,
        )
        txt = (
            self.txt_fc(outputs.text_embeds) if self.add_txt_fc else outputs.text_embeds
        )
        img = (
            self.img_fc(outputs.image_embeds)
            if self.add_img_fc
            else outputs.image_embeds
        )
        return txt, img
