import random
import scipy
import os
import numpy as np
import torch
import re
import math
import time


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
