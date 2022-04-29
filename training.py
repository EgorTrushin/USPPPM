#!/usr/bin/env python3
"""Training script for U.S. Patent Phrase to Phrase Matching."""

import gc
import math
import os
import random
import re
import time
import warnings

import numpy as np
import pandas as pd
import scipy as sp
import torch
import torch.nn as nn
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from src.meter import AverageMeter
from src.logger import get_logger

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

with open("config.yaml", "r") as file_obj:
    CFG = yaml.safe_load(file_obj)

if not os.path.exists(CFG["output_dir"]):
    os.makedirs(CFG["output_dir"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGGER = get_logger()

train = pd.read_csv(CFG["input_dir"] + "train.csv")
test = pd.read_csv(CFG["input_dir"] + "test.csv")
submission = pd.read_csv(CFG["input_dir"] + "sample_submission.csv")

# ====================================================
# Utils
# ====================================================
def get_score(y_true, y_pred):
    score = sp.stats.pearsonr(y_true, y_pred)[0]
    return score


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=42)


def get_cpc_texts():
    contexts = []
    pattern = "[A-Z]\d+"
    for file_name in os.listdir(CFG["input_dir"] + "cpc-data/CPCSchemeXML202105"):
        result = re.findall(pattern, file_name)
        if result:
            contexts.append(result)
    contexts = sorted(set(sum(contexts, [])))
    results = {}
    for cpc in ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]:
        with open(CFG["input_dir"] + f"cpc-data/CPCTitleList202202/cpc-section-{cpc}_20220201.txt") as f:
            s = f.read()
        pattern = f"{cpc}\t\t.+"
        result = re.findall(pattern, s)
        cpc_result = result[0].lstrip(pattern)
        for context in [c for c in contexts if c[0] == cpc]:
            pattern = f"{context}\t\t.+"
            result = re.findall(pattern, s)
            results[context] = cpc_result + ". " + result[0].lstrip(pattern)
    return results


cpc_texts = get_cpc_texts()
torch.save(cpc_texts, CFG["output_dir"] + "cpc_texts.pth")
train["context_text"] = train["context"].map(cpc_texts)
test["context_text"] = test["context"].map(cpc_texts)

train["text"] = train["anchor"] + "[SEP]" + train["target"] + "[SEP]" + train["context_text"]
test["text"] = test["anchor"] + "[SEP]" + test["target"] + "[SEP]" + test["context_text"]


# ====================================================
# CV split
# ====================================================
# train['score_map'] = train['score'].map({0.00: 0, 0.25: 1, 0.50: 2, 0.75: 3, 1.00: 4})
# Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
# for n, (train_index, val_index) in enumerate(Fold.split(train, train['score_map'])):
#    train.loc[val_index, 'fold'] = int(n)
# train['fold'] = train['fold'].astype(int)
dfx = pd.get_dummies(train, columns=["score"]).groupby(["anchor"], as_index=False).sum()
cols = [c for c in dfx.columns if c.startswith("score_") or c == "anchor"]
dfx = dfx[cols]

mskf = MultilabelStratifiedKFold(n_splits=CFG["n_fold"], shuffle=True, random_state=42)
labels = [c for c in dfx.columns if c != "anchor"]
dfx_labels = dfx[labels]
dfx["fold"] = -1

for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
    dfx.loc[val_, "fold"] = fold

train = train.merge(dfx[["anchor", "fold"]], on="anchor", how="left")


# ====================================================
# tokenizer
# ====================================================
tokenizer = AutoTokenizer.from_pretrained(CFG["model"])
tokenizer.save_pretrained(CFG["output_dir"] + "tokenizer/")
CFG["tokenizer"] = tokenizer
# tokenizer = AutoTokenizer.from_pretrained('tokenizer/')
# CFG["tokenizer"] = tokenizer


# ====================================================
# Define max_len
# ====================================================
lengths_dict = {}

lengths = []
tk0 = tqdm(cpc_texts.values(), total=len(cpc_texts))
for text in tk0:
    length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    lengths.append(length)
lengths_dict["context_text"] = lengths

for text_col in ["anchor", "target"]:
    lengths = []
    tk0 = tqdm(train[text_col].fillna("").values, total=len(train))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        lengths.append(length)
    lengths_dict[text_col] = lengths

CFG["max_len"] = (
    max(lengths_dict["anchor"]) + max(lengths_dict["target"]) + max(lengths_dict["context_text"]) + 4
)  # CLS + SEP + SEP + SEP
max_len = CFG["max_len"]
LOGGER.info(f"max_len: {max_len}")


# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text):
    inputs = cfg["tokenizer"](
        text, add_special_tokens=True, max_length=cfg["max_len"], padding="max_length", return_offsets_mapping=False
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df["text"].values
        self.labels = df["score"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label


# ====================================================
# Model
# ====================================================
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
        self.fc_dropout = nn.Dropout(cfg["fc_dropout"])
        self.fc = nn.Linear(self.config.hidden_size, self.cfg["target_size"])
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256), nn.Tanh(), nn.Linear(256, 1), nn.Softmax(dim=1)
        )
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
        # feature = torch.mean(last_hidden_states, 1)
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG["apex"])
    losses = AverageMeter()
    start = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG["apex"]):
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        if CFG["gradient_accumulation_steps"] > 1:
            loss = loss / CFG["gradient_accumulation_steps"]
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
        if (step + 1) % CFG["gradient_accumulation_steps"] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG["batch_scheduler"]:
                scheduler.step()
        if step % CFG["print_freq"] == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f}  "
                "LR: {lr:.8f}  ".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    grad_norm=grad_norm,
                    lr=scheduler.get_lr()[0],
                ),
                flush=True,
            )
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        if CFG["gradient_accumulation_steps"] > 1:
            loss = loss / CFG["gradient_accumulation_steps"]
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().to("cpu").numpy())
        if step % CFG["print_freq"] == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{0}/{1}]"
                "Elapsed {remain:s}"
                "Loss: {loss.val:.4f}({loss.avg:.4f})".format(
                    step, len(valid_loader), loss=losses, remain=timeSince(start, float(step + 1) / len(valid_loader))
                ),
                flush=True,
            )
    predictions = np.concatenate(preds)
    predictions = np.concatenate(predictions)
    return losses.avg, predictions


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to("cpu").numpy())
    predictions = np.concatenate(preds)
    return predictions


# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold):

    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds["fold"] != fold].reset_index(drop=True)
    valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)
    valid_labels = valid_folds["score"].values

    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, CFG["output_dir"] + "config.pth")
    model.to(device)

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": encoder_lr,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if "model" not in n],
                "lr": decoder_lr,
                "weight_decay": 0.0,
            },
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(
        model, encoder_lr=CFG["encoder_lr"], decoder_lr=CFG["decoder_lr"], weight_decay=CFG["weight_decay"]
    )
    optimizer = AdamW(
        optimizer_parameters, lr=CFG["encoder_lr"], eps=CFG["eps"], betas=(CFG["betas"][0], CFG["betas"][1])
    )

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg["scheduler"] == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg["num_warmup_steps"], num_training_steps=num_train_steps
            )
        elif cfg["scheduler"] == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg["num_warmup_steps"],
                num_training_steps=num_train_steps,
                num_cycles=cfg["num_cycles"],
            )
        return scheduler

    num_train_steps = int(len(train_folds) / CFG["batch_size"] * CFG["epochs"])
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    best_score = 0.0

    for epoch in range(CFG["epochs"]):

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)

        # scoring
        score = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time

        LOGGER.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        LOGGER.info(f"Epoch {epoch+1} - Score: {score:.4f}")

        if best_score < score:
            best_score = score
            LOGGER.info(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "predictions": predictions},
                CFG["output_dir"] + f"fold{fold}_best.pth",
            )

    predictions = torch.load(CFG["output_dir"] + f"fold{fold}_best.pth", map_location=torch.device("cpu"))[
        "predictions"
    ]
    valid_folds["pred"] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds


if __name__ == "__main__":

    def get_result(oof_df):
        labels = oof_df["score"].values
        preds = oof_df["pred"].values
        score = get_score(labels, preds)
        LOGGER.info(f"Score: {score:<.4f}")

    if CFG["train"]:
        oof_df = pd.DataFrame()
        for fold in range(CFG["n_fold"]):
            if fold in CFG["trn_fold"]:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info("========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(CFG["output_dir"] + "oof_df.pkl")
