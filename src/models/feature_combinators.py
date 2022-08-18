from typing import Callable, Dict

import torch


class LanguageAndVisionConcat(torch.nn.Module):
    def __init__(
        self,
        loss_fn,
        language_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
        language_module=None,
        vision_module=None,
        dual_module=None,
    ):
        super().__init__()
        if dual_module is None:
            self.language_module = language_module
            self.vision_module = vision_module
            self.dual_module = self._combine_modules(
                self.language_module, self.vision_module
            )
        else:
            self.dual_module = dual_module
        self.fusion = torch.nn.Linear(
            in_features=(language_feature_dim + vision_feature_dim),
            out_features=fusion_output_size,
        )
        self.fc = torch.nn.Linear(in_features=fusion_output_size, out_features=1)
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)

    def _combine_modules(self, language_module, vision_module) -> Callable:
        def dual_module(txt, img) -> Dict[str, torch.Tensor]:
            return {
                "txt": language_module(
                    torch.squeeze(txt, 1)
                ),  # added to avoid extra dim
                "img": vision_module(img),
            }

        return dual_module

    def forward(self, txt, img, label=None):  # TODO: test this None default
        dual_output = self.dual_module(txt, img)
        txt_features = torch.nn.functional.relu(dual_output["txt"])
        img_features = torch.nn.functional.relu(dual_output["img"])
        combined = torch.cat([txt_features, img_features], dim=1)
        fused = self.dropout(torch.nn.functional.relu(self.fusion(combined)))
        pred = self.fc(fused)
        loss = self.loss_fn(pred, label) if label is not None else label
        return (pred, loss)


class LanguageAndVisionWeightedImportance(torch.nn.Module):
    def __init__(
        self,
        language_feature_dim,
        vision_feature_dim,
        language_module=None,
        vision_module=None,
        dual_module=None,
    ):
        super().__init__()
        if dual_module is None:
            self.language_module = language_module
            self.vision_module = vision_module
            self.dual_module = self._combine_modules(
                self.language_module, self.vision_module
            )
        else:
            self.dual_module = dual_module
        self.fusion = torch.nn.Linear(
            in_features=(language_feature_dim + vision_feature_dim),
            out_features=2,
        )

    def _combine_modules(self, language_module, vision_module) -> Callable:
        def dual_module(txt, img) -> Dict[str, torch.Tensor]:
            return {
                "txt": language_module(
                    torch.squeeze(txt, 1)
                ),  # added to avoid extra dim
                "img": vision_module(img),
            }

        return dual_module

    def forward(self, txt, img):  # TODO: test this None default
        dual_output = self.dual_module(txt, img)
        txt_features = dual_output["txt"]
        img_features = dual_output["img"]
        combined = torch.cat([txt_features, img_features], dim=1)
        weights = torch.nn.functional.softmax(self.fusion(combined), dim=1)
        return {
            "txt": txt_features * weights[:, 0].unsqueeze(-1),
            "img": img_features * weights[:, 1].unsqueeze(-1),
        }
