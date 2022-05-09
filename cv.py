#!/usr/bin/env python3

import os

import pandas as pd
import scipy as sp
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold

INPUT_DIR = "/home/egortrushin/datasets/us-patent-phrase-to-phrase-matching/"


class CFG:
    fold_seed = 42
    n_fold = 5
    cv_scheme = 1
    trn_fold = [0, 1, 2, 3, 4]


def get_score(y_true, y_pred):
    score = sp.stats.pearsonr(y_true, y_pred)[0]
    return score


def get_result(oof_df):
    labels = oof_df["score"].values
    preds = oof_df["pred"].values
    score = get_score(labels, preds)
    return score


def train_loop(folds, fold, dir1):
    valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)
    predictions = torch.load(dir1 + f"fold{fold}_best.pth")["predictions"]
    valid_folds["pred"] = predictions
    return valid_folds


def get_oof_df(dir_):
    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            _oof_df = train_loop(train, fold, dir_)
            oof_df = pd.concat([oof_df, _oof_df])
            score = get_result(_oof_df)
            print(f"Fold {fold}= {score:<.4f}")
    oof_df = oof_df.reset_index(drop=True)
    return oof_df


train = pd.read_csv(INPUT_DIR + "train.csv")

if CFG.cv_scheme == 0:
    train["score_map"] = train["score"].map(
        {0.00: 0, 0.25: 1, 0.50: 2, 0.75: 3, 1.00: 4}
    )
    Fold = StratifiedKFold(
        n_splits=CFG.n_fold, shuffle=True, random_state=CFG.fold_seed
    )
    for n, (train_index, val_index) in enumerate(Fold.split(train, train["score_map"])):
        train.loc[val_index, "fold"] = int(n)
    train["fold"] = train["fold"].astype(int)
else:
    dfx = (
        pd.get_dummies(train, columns=["score"])
        .groupby(["anchor"], as_index=False)
        .sum()
    )
    cols = [c for c in dfx.columns if c.startswith("score_") or c == "anchor"]
    dfx = dfx[cols]
    mskf = MultilabelStratifiedKFold(
        n_splits=CFG.n_fold, shuffle=True, random_state=CFG.fold_seed
    )
    labels = [c for c in dfx.columns if c != "anchor"]
    dfx_labels = dfx[labels]
    dfx["fold"] = -1
    for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
        dfx.loc[val_, "fold"] = fold
    train = train.merge(dfx[["anchor", "fold"]], on="anchor", how="left")


if __name__ == "__main__":

    oof_df = get_oof_df("deberta_v3_large_bce_8319_838/ckpt/")
    score = get_result(oof_df)
    print(f"CV = {score:<.4f}")

    oof_df1 = get_oof_df("deberta_v3_large_rmse_8230/ckpt/")
    score = get_result(oof_df1)
    print(f"CV = {score:<.4f}")

    oof_df["pred"] = (oof_df["pred"] + oof_df1["pred"]) / 2.0
    score = get_result(oof_df)
    print(f"Ensemble CV = {score:<.4f}")
