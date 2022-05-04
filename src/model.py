import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification


class TransformerHead(nn.Module):
    def __init__(self, in_features, max_length, num_layers=1, nhead=8, num_targets=1):
        super().__init__()

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=in_features, nhead=nhead), num_layers=num_layers
        )
        self.row_fc = nn.Linear(in_features, 1)
        self.out_features = max_length

    def forward(self, x):
        out = self.transformer(x)
        out = self.row_fc(out).squeeze(-1)
        return out


class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg["model"], output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg["model"], config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        if cfg["head"] == "Transformer":
            self.feature_extractor = AutoModelForTokenClassification.from_pretrained(cfg["model"])
            in_features = self.feature_extractor.classifier.in_features
            self.attention = TransformerHead(
                in_features=in_features,
                max_length=cfg["max_len"],
                num_layers=cfg["transformer_head_layers"],
                nhead=cfg["nhead"],
                num_targets=1,
            )
            self.fc_dropout = nn.Dropout(cfg["fc_dropout"])
            self.fc = nn.Linear(self.attention.out_features, self.cfg["target_size"])
        else:
            self.attention = nn.Sequential(
                nn.Linear(self.config.hidden_size, cfg["att_hidden_size"]),
                nn.Tanh(),
                nn.Linear(cfg["att_hidden_size"], 1),
                nn.Softmax(dim=1),
            )
            self.fc_dropout = nn.Dropout(cfg["fc_dropout"])
            self.fc = nn.Linear(self.config.hidden_size, self.cfg["target_size"])

        self._init_weights(self.fc)
        self._init_weights(self.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        if self.cfg["head"] == "Transformer":
            feature = self.attention(last_hidden_states)
        else:
            weights = self.attention(last_hidden_states)
            feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output
