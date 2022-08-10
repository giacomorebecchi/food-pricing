import torch


class LanguageAndVisionConcat(torch.nn.Module):
    def __init__(
        self,
        loss_fn,
        language_module,
        vision_module,
        language_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
    ):
        super().__init__()
        self.language_module = language_module
        self.vision_module = vision_module
        self.fusion = torch.nn.Linear(
            in_features=(language_feature_dim + vision_feature_dim),
            out_features=fusion_output_size,
        )
        self.fc = torch.nn.Linear(in_features=fusion_output_size, out_features=1)
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, txt, img, label=None):  # TODO: test this None default
        txt_features = torch.nn.functional.relu(self.language_module(txt))
        img_features = torch.nn.functional.relu(self.vision_module(img))
        combined = torch.cat([txt_features, img_features], dim=1)
        fused = self.dropout(torch.nn.functional.relu(self.fusion(combined)))
        pred = self.fc(fused)
        loss = self.loss_fn(pred, label) if label is not None else label
        return (pred, loss)
