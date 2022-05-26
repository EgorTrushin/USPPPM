#!/usr/bin/env python3
"""Training script for U.S. Patent Phrase to Phrase Matching."""

import os
import re

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_cpc_texts(config):
    """
    Fix as provided by Nicholas Broad.

    https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/324928#1790476
    """
    contexts = []
    pattern = "[A-Z]\d+"
    for file_name in os.listdir(config["input_dir"] + "cpc-data/CPCSchemeXML202105"):
        result = re.findall(pattern, file_name)
        if result:
            contexts.append(result)
    contexts = sorted(set(sum(contexts, [])))
    results = {}
    for cpc in ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]:
        with open(config["input_dir"] + f"cpc-data/CPCTitleList202202/cpc-section-{cpc}_20220201.txt") as f:
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


def get_folds(df, config):
    """Fold splitting using MultilabelStratifiedKFold."""
    dfx = pd.get_dummies(df, columns=["score"]).groupby(["anchor"], as_index=False).sum()
    cols = [c for c in dfx.columns if c.startswith("score_") or c == "anchor"]
    dfx = dfx[cols]
    mskf = MultilabelStratifiedKFold(n_splits=config["n_fold"], shuffle=True, random_state=config["fold_seed"])
    labels = [c for c in dfx.columns if c != "anchor"]
    dfx_labels = dfx[labels]
    dfx["fold"] = -1
    for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
        dfx.loc[val_, "fold"] = fold
    df = df.merge(dfx[["anchor", "fold"]], on="anchor", how="left")
    return df


def get_max_len(train, cpc_texts, tokenizer):
    """Determine max_len."""
    lengths_dict = {}
    lengths = []
    tk0 = tqdm(cpc_texts.values(), total=len(cpc_texts), disable=True)
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        lengths.append(length)
    lengths_dict["context_text"] = lengths

    for text_col in ["anchor", "target"]:
        lengths = []
        tk0 = tqdm(train[text_col].fillna("").values, total=len(train), disable=True)
        for text in tk0:
            length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            lengths.append(length)
        lengths_dict[text_col] = lengths

    max_len = (
        max(lengths_dict["anchor"]) + max(lengths_dict["target"]) + max(lengths_dict["context_text"]) + 4
    )  # CLS + SEP + SEP + SEP
    return max_len


def get_pearson_score(y_true, y_pred):
    """Calculate Pearson correlation."""
    score = pearsonr(y_true, y_pred)[0]
    return score


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

        self.model = AutoModel.from_pretrained(model_name, self.config)

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
        config.hidden_dropout_prob = hparams["fc_dropout"]
        self.base = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    def forward(self, inputs):
        base_output = self.base(**inputs)
        return base_output[0]


class AbhishekModel(nn.Module):
    def __init__(self, model_name, hparams):
        super().__init__()
        self.model_name = model_name

        config = AutoConfig.from_pretrained(model_name)
        config.update(
            {
                "output_hidden_states": True,
                "add_pooling_layer": True,
                "num_labels": 1,
            }
        )
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output = nn.Linear(config.hidden_size, 1)

    def forward(self, inputs):
        transformer_out = self.transformer(**inputs)
        output = transformer_out.pooler_output
        output = self.dropout(output)
        output = self.output(output)
        return output


loss_dict = {
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
    "MSELoss": nn.MSELoss(),
}


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
        elif self.hparams.model_name == "AbhishekModel":
            self.model = AbhishekModel(self.hparams.base_model_name, self.hparams.model_hparams)
        else:
            assert False, f'Unknown model_name: "{self.hparams.model_name}"'
        # Create loss
        self.loss = loss_dict[self.hparams.loss_name]

    def forward(self, text, mask):
        """Forward."""
        return self.model(text, mask)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        if self.hparams.optimizer_name == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # total_steps = len(self.trainer._data_connector._train_dataloader_source.dataloader()) * self.trainer.max_epochs
        total_steps = len(self.train_dataloader()) * self.trainer.max_epochs
        if self.hparams.scheduler_name == "linear_schedule_with_warmup":
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps,
            )
        elif self.hparams.scheduler_name == "cosine_schedule_with_warmup":
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps,
            )
        else:
            assert False, f'Unknown scheduler: "{self.hparams.scheduler_name}"'

        lr_scheduler_dict = {"scheduler": self.scheduler, "interval": "step"}
        return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_dict}

    def training_step(self, batch, batch_idx):
        """Training."""
        inputs, labels = batch[0], batch[1]
        preds = self.model(inputs)
        loss = self.loss(preds.squeeze(1), labels)
        self.log("trn_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Calculate validation scores/metrics."""
        inputs, labels = batch[0], batch[1]
        preds = self.model(inputs)
        loss = self.loss(preds.squeeze(1), labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"preds": preds.squeeze(dim=-1), "labels": batch[1].squeeze(dim=-1)}

    def validation_epoch_end(self, outs):
        """Calculate Pearson score at the end of validation."""
        preds = []
        labels = []
        for out in outs:
            preds.append(out["preds"])
            labels.append(out["labels"])
        preds = torch.cat(preds).cpu().numpy().flatten()
        labels = torch.cat(labels).cpu().numpy().flatten()
        score = get_pearson_score(labels, preds)
        self.log("val_score", score, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        """Prediction."""
        preds = self.model(batch)
        return preds


if __name__ == "__main__":
    with open("config.yaml", "r") as file_obj:
        config = yaml.safe_load(file_obj)

    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])

    df = pd.read_csv(config["input_dir"] + "train.csv")
    cpc_texts = get_cpc_texts(config)
    df["context_text"] = df["context"].map(cpc_texts)
    if config["context_text_lower"]:
        df["text"] = df["anchor"] + "[SEP]" + df["target"] + "[SEP]" + df["context_text"].apply(str.lower)
    else:
        df["text"] = df["anchor"] + "[SEP]" + df["target"] + "[SEP]" + df["context_text"]

    df = get_folds(df, config)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_name"])
    max_len = get_max_len(df, cpc_texts, tokenizer)
    config["max_len"] = max_len
    print("max_len:", max_len)

    pl.seed_everything(config["seed"])

    for fold in range(config["n_fold"]):
        if fold in config["trn_fold"]:
            print(f"#### Fold {fold} training")

            train_df = df[df["fold"] != fold].reset_index(drop=True)
            valid_df = df[df["fold"] == fold].reset_index(drop=True)

            train_dataset = PhraseSimilarityDataset(train_df, tokenizer, config["max_len"])
            val_dataset = PhraseSimilarityDataset(valid_df, tokenizer, config["max_len"])

            train_dataloader = DataLoader(
                train_dataset, batch_size=config["trn_batch_size"], num_workers=config["num_workers"], shuffle=True
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=config["val_batch_size"], num_workers=config["num_workers"], shuffle=False
            )

            filename = f"model-f{fold}-{{val_score:.4f}}"
            checkpoint_callback = ModelCheckpoint(
                save_weights_only=True,
                monitor="val_score",
                dirpath=config["output_dir"],
                mode="max",
                filename=filename,
                verbose=1,
            )

            trainer = pl.Trainer(
                gpus=-1 if torch.cuda.is_available() else 0,
                logger=None,
                callbacks=[checkpoint_callback],
                **config["trainer"],
            )

            driver = PhraseSimilarityModel(**config["model"])

            trainer.fit(driver, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
