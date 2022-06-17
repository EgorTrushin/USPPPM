#!/usr/bin/env python3
"""Prediction script for U.S. Patent Phrase to Phrase Matching."""

import os
import re
import glob
import gc
import scipy

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from scipy.stats import trim_mean

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_cpc_texts(config):
    contexts = []
    pattern = "[A-Z]\d+"
    for file_name in os.listdir("../input/uspppm-tools/cpc-data/CPCSchemeXML202105"):
        result = re.findall(pattern, file_name)
        if result:
            contexts.append(result)
    contexts = sorted(set(sum(contexts, [])))
    results = {}
    for cpc in ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]:
        with open("../input/uspppm-tools/" + f"cpc-data/CPCTitleList202202/cpc-section-{cpc}_20220201.txt") as f:
            s = f.read()
        pattern = f"{cpc}\t\t.+"
        result = re.findall(pattern, s)
        pattern = "^" + pattern[:-2]
        cpc_result = re.sub(pattern, "", result[0])
        for context in [c for c in contexts if c[0] == cpc]:
            pattern = f"{context}\t\t.+"
            result = re.findall(pattern, s)
            pattern = "^" + pattern[:-2]
            results[context] = cpc_result + ". " + re.sub(pattern, "", result[0])
    return results


def prepare_input(text, tokenizer, max_len):
    """Prepare input for Transformer."""
    inputs = tokenizer(
        text, add_special_tokens=True, max_length=max_len, padding="max_length", return_offsets_mapping=False
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class PhraseSimilarityDataset(Dataset):
    """Dataset."""

    def __init__(self, df, tokenizer, max_len, with_labels=True):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.with_labels = with_labels

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        inputs = prepare_input(self.df.text[index], self.tokenizer, self.max_len)
        if self.with_labels:
            label = torch.tensor(self.df.score.iloc[index], dtype=torch.float)
            return inputs, label
        else:
            return inputs


class NakamaModel(nn.Module):
    """Model from Nakama notebook."""

    def __init__(self, model_name, hparams, pretrained=True):
        """Initialize model."""
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)

        self.model = AutoModel.from_config(config=self.config)

        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, hparams["att_hidden_size"]),
            nn.Tanh(),
            nn.Linear(hparams["att_hidden_size"], 1),
            nn.Softmax(dim=1),
        )
        self.fc_dropout = nn.Dropout(hparams["fc_dropout"])
        self.fc = nn.Linear(self.config.hidden_size, hparams["target_size"])

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
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output


class SimpleModel(nn.Module):
    def __init__(self, model_name, hparams):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 1
        self.base = AutoModelForSequenceClassification.from_config(config=config)

    def forward(self, inputs):
        base_output = self.base(**inputs)

        return base_output[0]


class PhraseSimilarityModel(pl.LightningModule):
    """Lightning Module."""

    def __init__(
        self, model_name, base_model_name, model_hparams, loss_name, optimizer_name, optimizer_hparams, scheduler_name
    ):
        """Initialize lightning model."""
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        if self.hparams.model_name == "NakamaModel":
            self.model = NakamaModel(self.hparams.base_model_name, self.hparams.model_hparams)
        elif self.hparams.model_name == "SimpleModel":
            self.model = SimpleModel(self.hparams.base_model_name, self.hparams.model_hparams)
        else:
            assert False, f'Unknown model_name: "{self.hparams.model_name}"'

    def forward(self, text, mask):
        """Forward."""
        return self.model(text, mask)

    def predict_step(self, batch, batch_idx):
        """Prediction."""
        preds = self.model(batch)
        return preds


def make_predictions(ckpt_dir, config, predict_dataloader, l_mean=True):
    all_predictions = []
    for model in glob.glob(ckpt_dir + "/*.ckpt"):

        trainer = pl.Trainer(
            gpus=-1 if torch.cuda.is_available() else 0,
            logger=None,
            **config["trainer"],
        )

        driver = PhraseSimilarityModel.load_from_checkpoint(model, **config["model"])

        predictions = trainer.predict(driver, predict_dataloader)

        preds = []
        for batch in predictions:
            preds += batch.squeeze(1).tolist()

        np_preds = np.array(preds)
        scaled_preds = (np_preds - np_preds.min()) / (np_preds.max() - np_preds.min())
        all_predictions.append(scaled_preds)

        del driver, trainer
        torch.cuda.empty_cache()
        gc.collect()

    if l_mean:
        final_predictions = np.array(all_predictions).mean(axis=0)
    else:
        final_predictions = np.array(all_predictions)
    return final_predictions


def postprocess(inputs, axis=0, spread_lim=0.1, proportiontocut=0.175, verb=True):
    """When spead of provided predictions is smaller than spread_lim, use usual mean,
    when spead of provided predictions is larger than spread_lim, use trimmed mean
    with provided proportiontocut."""
    spread = inputs.max(axis=axis) - inputs.min(axis=axis)
    return np.where(
        spread < spread_lim, np.mean(inputs, axis=axis), scipy.stats.trim_mean(inputs, proportiontocut, axis=axis)
    )


input = ("/home/egortrushin/GitHub/USPPPM_models",)  # "../input/"
tools = "../input/uspppm-tools/"
models = {
    "deberta-v3-large SimpleModel PearsonLoss": {
        "path": input + "deberta-v3-large-08357",
        "path2": tools + "deberta-v3-large",
        "max_len": 133,
    },  # LB=0.8398
    "deberta-v3-large NakamaModel PearsonLoss": {
        "path": input + "deberta-v3-large-08346",
        "path2": tools + "deberta-v3-large",
        "max_len": 133,
    },  # LB=0.8393
    "bert_for_patents SimpleModel PearsonLoss": {
        "path": input + "bert-for-patents-08232",
        "path2": tools + "bert-for-patents",
        "max_len": 115,
    },  # LB=0.8365
    "bert_for_patents NakamaModel PearsonLoss": {
        "path": input + "bert-for-patents-08258",
        "path2": tools + "bert-for-patents",
        "max_len": 115,
    },
    "albert_xxlarge_v2 SimpleModel PearsonLoss": {
        "path": input + "albert-xxlarge-v2-08167",
        "path2": tools + "albert-xxlarge-v2",
        "max_len": 124,
    },
    "electra_large SimpleModel PearsonLoss": {
        "path": input + "electra-large-08297",
        "path2": tools + "electra-large",
        "max_len": 122,
    },
    "electra_large SimpleModel MSELoss": {
        "path": input + "electra-large-08285",
        "path2": tools + "electra-large",
        "max_len": 122,
    },
    "funnel_xlarge SimpleModel PearsonLoss": {
        "path": input + "funnel-xlarge-08275",
        "path2": tools + "funnel-xlarge",
        "max_len": 122,
    },
    "BioM-ELECTRA-Large SimpleModel PearsonLoss": {
        "path": input + "biom-electra-large-discriminator-08142",
        "path2": tools + "BioM-ELECTRA-Large-Discriminator",
        "max_len": 108,
    },
    # "funnel_large SimpleModel PearsonLoss": {
    #     "path": input + "funnel-large-08213",
    #     "path2": tools + "funnel-large",
    #     "max_len": 122,
    # },
    # "funnel_xlarge_base SimpleModel PearsonLoss": {
    #     "path": input + "funnel-xlarge-base-08285",
    #     "path2": tools + "funnel-xlarge-base",
    #     "max_len": 122,
    # },
    # "deberta-v3-large SimpleModel PearsonLoss Pseudo-Labelling": {
    #     "path": input + "deberta-v3-large-08380",
    #     "path2": tools + "deberta-v3-large",
    #     "max_len": 133,
    # },  # LB=0.8406
}

all_predictions = list()
for model in models:
    print(model)
    ckpt_dir = models[model]["path"]
    with open(ckpt_dir + "/config.yaml", "r") as file_obj:
        config = yaml.safe_load(file_obj)
    config["input_dir"] = "../input/us-patent-phrase-to-phrase-matching/"
    config["model"]["base_model_name"] = models[model]["path2"]
    config["max_len"] = models[model]["max_len"]
    config["num_workers"] = 1
    config["val_batch_size"] = 64

    predict_dataloader = get_dataloader(config)
    predictions = make_predictions(ckpt_dir, config, predict_dataloader, l_mean=False)
    all_predictions.append(predictions)

    submission_csv = pd.read_csv(config["input_dir"] + "sample_submission.csv")
    submission_csv["score"] = final_predictions
    submission_csv.to_csv("submission.csv", index=False)
