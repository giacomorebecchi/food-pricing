from .base_model import FoodPricingBaseModel
from .dual_encoding.pretrained_clip import PreTrainedCLIP


class FPCLIPConcatModel(FoodPricingBaseModel):
    def __init__(self, *args, **kwargs):
        super(FPCLIPConcatModel, self).__init__(*args, **kwargs)

    def _build_dual_module(self) -> PreTrainedCLIP:
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
        self._update_clip_hparams(
            processor_config=clip.processor.feature_extractor,
            encoder_features=clip.encoder_features,
        )
        return clip

    def _update_clip_hparams(self, processor_config, encoder_features: int) -> None:
        self.hparams.update(
            {
                "img_dim": processor_config.crop_size,
                "img_mean": processor_config.image_mean,
                "img_std": processor_config.image_std,
            }
        )
        self.hparams.update(
            {
                "projection_dim": encoder_features,
            }
        )

    def _add_model_specific_hparams(self) -> None:
        model_specific_hparams = {
            "dual_module": True,
            "clip_model": "clip-italian/clip-italian",
            "processor_clip_model": self.hparams.get(
                "clip_model", "clip-italian/clip-italian"
            ),  # Default is same as "clip_model" param
        }
        self.hparams.update({**model_specific_hparams, **self.hparams})


class FPCLIPWeightedConcatModel(FPCLIPConcatModel):
    def __init__(self, *args, **kwargs):
        super(FPCLIPWeightedConcatModel, self).__init__(*args, **kwargs)

    def _add_model_specific_hparams(self) -> None:
        model_specific_hparams = {
            "dual_module": True,
            "attention_module": True,
            "clip_model": "clip-italian/clip-italian",
            "processor_clip_model": self.hparams.get(
                "clip_model", "clip-italian/clip-italian"
            ),  # Default is same as "clip_model" param
        }
        self.hparams.update({**model_specific_hparams, **self.hparams})
