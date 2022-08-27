from typing import Tuple

from torch import Tensor, cat, nn


class LanguageAndVisionConcat(nn.Module):
    def __init__(
        self,
        language_feature_dim: int,
        vision_feature_dim: int,
        fusion_output_dim: int,
        dropout_p: float,
    ) -> None:
        super(LanguageAndVisionConcat, self).__init__()
        self.fusion = nn.Sequential(
            nn.LayerNorm(language_feature_dim + vision_feature_dim),
            nn.Linear(
                in_features=(language_feature_dim + vision_feature_dim),
                out_features=fusion_output_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(in_features=fusion_output_dim, out_features=1),
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
