import random
import scipy
import os
import numpy as np
import pandas as pd
import torch
import re
import math
import time
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm


def get_folds(train, CFG):
    if CFG["cv_scheme"] == 0:
        train["score_map"] = train["score"].map({0.00: 0, 0.25: 1, 0.50: 2, 0.75: 3, 1.00: 4})
        Fold = StratifiedKFold(n_splits=CFG["n_fold"], shuffle=True, random_state=CFG["fold_seed"])
        for n, (train_index, val_index) in enumerate(Fold.split(train, train["score_map"])):
            train.loc[val_index, "fold"] = int(n)
        train["fold"] = train["fold"].astype(int)
    else:
        dfx = pd.get_dummies(train, columns=["score"]).groupby(["anchor"], as_index=False).sum()
        cols = [c for c in dfx.columns if c.startswith("score_") or c == "anchor"]
        dfx = dfx[cols]
        mskf = MultilabelStratifiedKFold(n_splits=CFG["n_fold"], shuffle=True, random_state=CFG["fold_seed"])
        labels = [c for c in dfx.columns if c != "anchor"]
        dfx_labels = dfx[labels]
        dfx["fold"] = -1
        for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
            dfx.loc[val_, "fold"] = fold
        train = train.merge(dfx[["anchor", "fold"]], on="anchor", how="left")
    return train


def get_max_len(train, cpc_texts, tokenizer):
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


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_score(y_true, y_pred):
    score = scipy.stats.pearsonr(y_true, y_pred)[0]
    return score


def get_cpc_texts(CFG):
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


def get_result(oof_df):
    labels = oof_df["score"].values
    preds = oof_df["pred"].values
    score = get_score(labels, preds)
    return score
