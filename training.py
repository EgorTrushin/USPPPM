#!/usr/bin/env python3
"""Training script for U.S. Patent Phrase to Phrase Matching."""

import os
import warnings

import pandas as pd
import torch
import yaml
from src.logger import get_logger
from src.train import train_loop
from src.utils import get_cpc_texts, get_folds, get_max_len, get_result, seed_everything
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"


if __name__ == "__main__":
    with open("config.yaml", "r") as file_obj:
        CFG = yaml.safe_load(file_obj)

    if not os.path.exists(CFG["output_dir"]):
        os.makedirs(CFG["output_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(seed=CFG["seed"])

    LOGGER = get_logger(CFG["output_dir"] + "train")

    train = pd.read_csv(CFG["input_dir"] + "train.csv")
    # test = pd.read_csv(CFG["input_dir"] + "test.csv")
    submission = pd.read_csv(CFG["input_dir"] + "sample_submission.csv")

    cpc_texts = get_cpc_texts(CFG)
    torch.save(cpc_texts, CFG["output_dir"] + "cpc_texts.pth")
    train["context_text"] = train["context"].map(cpc_texts)

    train["text"] = train["anchor"] + "[SEP]" + train["target"] + "[SEP]" + train["context_text"]

    train = get_folds(train, CFG)

    tokenizer = AutoTokenizer.from_pretrained(CFG["model"])
    tokenizer.save_pretrained(CFG["output_dir"] + "tokenizer/")
    CFG["tokenizer"] = tokenizer

    max_len = get_max_len(train, cpc_texts, tokenizer)
    CFG["max_len"] = max_len
    LOGGER.info(f"max_len: {max_len}")

    oof_df = pd.DataFrame()
    for fold in range(CFG["n_fold"]):
        if fold in CFG["trn_fold"]:
            _oof_df = train_loop(train, fold, device, LOGGER, CFG)
            oof_df = pd.concat([oof_df, _oof_df])
            score = get_result(_oof_df)
            LOGGER.info(f"#### Fold {fold} result: {score:<.4f}")
    oof_df = oof_df.reset_index(drop=True)
    LOGGER.info("#### CV")
    score = get_result(oof_df)
    LOGGER.info(f"Score: {score:<.4f}")
    oof_df.to_pickle(CFG["output_dir"] + "oof_df.pkl")
