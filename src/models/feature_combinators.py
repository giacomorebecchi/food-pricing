import logging
from typing import List, Tuple, Union

from torch import Tensor, cat, nn


class LanguageAndVisionConcat(nn.Module):
    def __init__(
        self,
        language_feature_dim: int,
        vision_feature_dim: int,
        fusion_output_dim: Union[int, List[int]],
        dropout_p: float,
    ) -> None:
        super(LanguageAndVisionConcat, self).__init__()
        if isinstance(fusion_output_dim, list):
            self.fusion_output_dim = fusion_output_dim
        else:
            self.fusion_output_dim = [fusion_output_dim]

        layers = []
        in_features = language_feature_dim + vision_feature_dim
        for dim in self.fusion_output_dim:
            try:
                assert isinstance(dim, int)
            except AssertionError:
                logging.error(
                    "Parameter 'fusion_output_dim' is neither a List of integers "
                    "nor a single integer"
                )
            layers.extend(
                [
                    nn.Linear(in_features=in_features, out_features=dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_p),
                ]
            )
            in_features = dim

        self.fusion = nn.Sequential(
            nn.LayerNorm(language_feature_dim + vision_feature_dim),
            *layers,
            nn.Linear(in_features=in_features, out_features=1),
        )

    def forward(self, txt: Tensor, img: Tensor) -> Tensor:
        combined = cat([txt, img], dim=1)
        pred = self.fusion(combined)
        return pred


class LanguageAndVisionWeightedImportance(nn.Module):
    def __init__(
        self,
        language_feature_dim: int,
        vision_feature_dim: int,
    ) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(
                in_features=(language_feature_dim + vision_feature_dim),
                out_features=2,
            ),
            nn.Softmax(dim=1),
        )

    def forward(self, txt: Tensor, img: Tensor) -> Tuple[Tensor, Tensor]:
        combined = cat([txt, img], dim=1)
        weights: Tensor = self.fusion(combined)
        return (
            txt * weights[:, 0].unsqueeze(-1),  # txt
            img * weights[:, 1].unsqueeze(-1),  # img
        )
