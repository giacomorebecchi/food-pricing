import torch

from .base_model import FoodPricingBaseModel
from .dual_encoding.pretrained_clip import PreTrainedCLIP
from .feature_combinators import LanguageAndVisionConcat


class FPCLIPConcatModel(FoodPricingBaseModel):
    def __init__(self, *args, **kwargs):
        super(FPCLIPConcatModel, self).__init__(*args, **kwargs)

    def _build_dual_transform(self):
        model_kwargs = {"pretrained_model_name_or_path": self.hparams.clip_model}
        processor_kwargs = {
            "pretrained_model_name_or_path": self.hparams.processor_clip_model
        }
        self.clip = PreTrainedCLIP(
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
            img_feature_dim=self.hparams.language_feature_dim,
            txt_feature_dim=self.hparams.vision_feature_dim,
            return_tensors=None,
        )
        self._update_clip_hparams()

    def _update_clip_hparams(self):
        processor_config = self.clip.processor.feature_extractor
        self.hparams.update(
            {
                "img_dim": processor_config.crop_size,
                "img_mean": processor_config.image_mean,
                "img_std": processor_config.image_std,
            }
        )
        self.hparams.update(
            {
                "projection_dim": self.clip.encoder_features,
            }
        )

    def _build_model(self):
        return LanguageAndVisionConcat(
            loss_fn=torch.nn.MSELoss(),
            dual_module=self.clip,
            language_feature_dim=self.hparams.language_feature_dim,
            vision_feature_dim=self.hparams.vision_feature_dim,
            fusion_output_size=self.hparams.fusion_output_size,
            dropout_p=self.hparams.dropout_p,
        )

    def _add_model_specific_hparams(self) -> None:
        model_specific_hparams = {
            "dual_model": True,
            "dropout_p": 0.2,
            "fusion_output_size": 512,
            "clip_model": "clip-italian/clip-italian",
            "processor_clip_model": self.hparams.get(
                "clip_model", "clip-italian/clip-italian"
            ),  # Default is same as "clip_model" param
        }
        self.hparams.update({**model_specific_hparams, **self.hparams})
