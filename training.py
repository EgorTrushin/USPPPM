#!/usr/bin/env python3
"""Training script for U.S. Patent Phrase to Phrase Matching."""

import gc
import os
import time
import warnings

import pandas as pd
import torch
import torch.nn as nn
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from src.logger import get_logger
from src.utils import seed_everything, get_cpc_texts, get_score, get_result
from src.data import TrainDataset
from src.model import CustomModel
from src.train import train_fn
from src.valid import valid_fn

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

with open("config.yaml", "r") as file_obj:
    CFG = yaml.safe_load(file_obj)

if not os.path.exists(CFG["output_dir"]):
    os.makedirs(CFG["output_dir"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything(seed=CFG["seed"])

LOGGER = get_logger(CFG["output_dir"] + "train")

train = pd.read_csv(CFG["input_dir"] + "train.csv")
test = pd.read_csv(CFG["input_dir"] + "test.csv")
submission = pd.read_csv(CFG["input_dir"] + "sample_submission.csv")

cpc_texts = get_cpc_texts(CFG)
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

CFG["max_len"] = (
    max(lengths_dict["anchor"]) + max(lengths_dict["target"]) + max(lengths_dict["context_text"]) + 4
)  # CLS + SEP + SEP + SEP
max_len = CFG["max_len"]
LOGGER.info(f"max_len: {max_len}")


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

        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, CFG)
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device, CFG)

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
        score = get_result(oof_df)
        LOGGER.info(f"Score: {score:<.4f}")
        oof_df.to_pickle(CFG["output_dir"] + "oof_df.pkl")
