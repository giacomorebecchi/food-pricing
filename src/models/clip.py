import torch
from transformers import AutoProcessor

from .base_model import FoodPricingBaseModel
from .dual_encoding.pretrained_clip import PreTrainedCLIP
from .feature_combinators import LanguageAndVisionConcat


class FPCLIPConcatModel(FoodPricingBaseModel):
    def __init__(self, *args, **kwargs):
        super(FPCLIPConcatModel, self).__init__(*args, **kwargs)

    def _build_model(self):
        model_kwargs = {"pretrained_model_name_or_path": self.hparams.clip_model}
        processor_kwargs = {
            "pretrained_model_name_or_path": self.hparams.processor_clip_model
        }
        clip = PreTrainedCLIP(
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
            img_feature_dim=self.hparams.language_feature_dim,
            txt_feature_dim=self.hparams.vision_feature_dim,
            return_tensors=None,
        )
        self.hparams.update(
            {
                "projection_dim": clip.encoder_features,
            }
        )
        return LanguageAndVisionConcat(
            loss_fn=torch.nn.MSELoss(),
            dual_module=clip,
            language_feature_dim=self.hparams.language_feature_dim,
            vision_feature_dim=self.hparams.vision_feature_dim,
            fusion_output_size=self.hparams.fusion_output_size,
            dropout_p=self.hparams.dropout_p,
        )

    def _add_model_specific_hparams(self) -> None:
        model_specific_hparams = {
            "dropout_p": 0.2,
            "fusion_output_size": 512,
            "clip_model": "clip-italian/clip-italian",
            "processor_clip_model": self.hparams.get(
                "clip_model", "clip-italian/clip-italian"
            ),  # Default is same as "clip_model" param
        }
        self.hparams.update({**model_specific_hparams, **self.hparams})
        # Once we have the correct name of the CLIP Model, we need to
        # retrieve the image mean and std and set them as parameters
        # before initializing all the other objects in the class
        processor_config = AutoProcessor.from_pretrained(
            self.hparams.processor_clip_model
        ).feature_extractor
        self.hparams.update(
            {
                "img_dim": processor_config.crop_size,
                "img_mean": processor_config.image_mean,
                "img_std": processor_config.image_std,
            }
        )
