#!/usr/bin/env python3
"""Training script for U.S. Patent Phrase to Phrase Matching."""

import os
import re

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
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

        self.model = AutoModel.from_pretrained(model_name, config=self.config)

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


class MDropModel(nn.Module):
    def __init__(self, model_name, hparams):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name)
        config.output_hidden_states = True
        self.base = AutoModel.from_pretrained(model_name, config=config)
        dim = config.hidden_size
        self.dropout_0 = nn.Dropout(p=0)
        self.dropout_1 = nn.Dropout(p=0.1)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_3 = nn.Dropout(p=0.3)
        self.dropout_4 = nn.Dropout(p=0.4)
        self.cls = nn.Linear(dim, 1)

    def forward(self, inputs):
        base_output = self.base(**inputs)
        output = base_output.hidden_states[-1]
        output_0 = self.cls(self.dropout_0(torch.mean(output, dim=1)))
        output_1 = self.cls(self.dropout_1(torch.mean(output, dim=1)))
        output_2 = self.cls(self.dropout_2(torch.mean(output, dim=1)))
        output_3 = self.cls(self.dropout_3(torch.mean(output, dim=1)))
        output_4 = self.cls(self.dropout_4(torch.mean(output, dim=1)))
        output = torch.mean(torch.stack([output_0, output_1, output_2, output_3, output_4], dim=0), dim=0)
        return output


class WKPooling(nn.Module):
    def __init__(self, layer_start: int = 4, context_window_size: int = 2):
        super(WKPooling, self).__init__()
        self.layer_start = layer_start
        self.context_window_size = context_window_size

    def forward(self, all_hidden_states, batch):

        ft_all_layers = all_hidden_states
        org_device = ft_all_layers.device
        all_layer_embedding = ft_all_layers.transpose(1, 0)
        all_layer_embedding = all_layer_embedding[:, self.layer_start :, :, :]  # Start from 4th layers output
        # torch.qr is slow on GPU (see https://github.com/pytorch/pytorch/issues/22573). So compute it on CPU until issue is fixed
        all_layer_embedding = all_layer_embedding.cpu()

        attention_mask = batch["token_mask"].cpu().numpy()
        unmask_num = np.array([sum(mask) for mask in attention_mask]) - 1  # Not considering the last item
        embedding = []
        # One sentence at a time
        for sent_index in range(len(unmask_num)):
            sentence_feature = all_layer_embedding[sent_index, :, : unmask_num[sent_index], :]
            one_sentence_embedding = []
            # Process each token
            for token_index in range(sentence_feature.shape[1]):
                token_feature = sentence_feature[:, token_index, :]
                # 'Unified Word Representation'
                token_embedding = self.unify_token(token_feature)
                one_sentence_embedding.append(token_embedding)

            one_sentence_embedding = torch.stack(one_sentence_embedding)
            sentence_embedding = self.unify_sentence(sentence_feature, one_sentence_embedding)
            embedding.append(sentence_embedding)

        output_vector = torch.stack(embedding).to(org_device)
        return output_vector

    def unify_token(self, token_feature):
        ## Unify Token Representation
        window_size = self.context_window_size

        alpha_alignment = torch.zeros(token_feature.size()[0], device=token_feature.device)
        alpha_novelty = torch.zeros(token_feature.size()[0], device=token_feature.device)

        for k in range(token_feature.size()[0]):
            left_window = token_feature[k - window_size : k, :]
            right_window = token_feature[k + 1 : k + window_size + 1, :]
            window_matrix = torch.cat([left_window, right_window, token_feature[k, :][None, :]])
            Q, R = torch.qr(window_matrix.T)

            r = R[:, -1]
            alpha_alignment[k] = torch.mean(self.norm_vector(R[:-1, :-1], dim=0), dim=1).matmul(
                R[:-1, -1]
            ) / torch.norm(r[:-1])
            alpha_alignment[k] = 1 / (alpha_alignment[k] * window_matrix.size()[0] * 2)
            alpha_novelty[k] = torch.abs(r[-1]) / torch.norm(r)

        # Sum Norm
        alpha_alignment = alpha_alignment / torch.sum(alpha_alignment)  # Normalization Choice
        alpha_novelty = alpha_novelty / torch.sum(alpha_novelty)

        alpha = alpha_novelty + alpha_alignment
        alpha = alpha / torch.sum(alpha)  # Normalize

        out_embedding = torch.mv(token_feature.t(), alpha)
        return out_embedding

    def norm_vector(self, vec, p=2, dim=0):
        # Implements the normalize() function from sklearn
        vec_norm = torch.norm(vec, p=p, dim=dim)
        return vec.div(vec_norm.expand_as(vec))

    def unify_sentence(self, sentence_feature, one_sentence_embedding):
        # Unify Sentence By Token Importance
        sent_len = one_sentence_embedding.size()[0]

        var_token = torch.zeros(sent_len, device=one_sentence_embedding.device)
        for token_index in range(sent_len):
            token_feature = sentence_feature[:, token_index, :]
            sim_map = self.cosine_similarity_torch(token_feature)
            var_token[token_index] = torch.var(sim_map.diagonal(-1))

        var_token = var_token / torch.sum(var_token)
        sentence_embedding = torch.mv(one_sentence_embedding.t(), var_token)

        return sentence_embedding

    def cosine_similarity_torch(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


class WKPoolingModel(nn.Module):
    def __init__(self, model_name, hparams, pretrained=True):
        """Initialize model."""
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)

        self.model = AutoModel.from_pretrained(model_name, config=self.config)

        self.fc_dropout = nn.Dropout(hparams["fc_dropout"])
        self.fc = nn.Linear(self.config.hidden_size, hparams["target_size"])
        self.wkpool = WKPooling(layer_start=4)
        self._init_weights(self.wkpool)
        self._init_weights(self.fc)

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

    def forward(self, inputs):
        outputs = self.model(input_ids=inputs["token_ids"], attention_mask=inputs["mask"])
        all_hidden_states = torch.stack(outputs[1])
        wkpooling_embeddings = self.wkpool(all_hidden_states, inputs)
        logits = self.fc(wkpooling_embeddings)  # regression head
        return logits


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MeanPoolingModel(nn.Module):
    def __init__(self, model_name, hparams):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name)
        self.base = AutoModel.from_pretrained(model_name, config=config)
        self.drop = nn.Dropout(p=hparams["fc_dropout"])
        self.pooler = MeanPooling()
        self.fc = nn.Linear(config.hidden_size, 1)

    def forward(self, inputs):
        # base_output = self.base(**inputs)
        # return base_output[0]
        out = self.base(input_ids=inputs["token_id"], attention_mask=inputs["token_mask"], output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, inputs["token_mask"])
        out = self.drop(out)
        outputs = self.fc(out)
        return outputs


class PearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        x = outputs - outputs.mean()
        y = targets - targets.mean()
        first = x / torch.linalg.norm(x)
        second = y / torch.linalg.norm(y)
        r = (first * second).sum()
        return -r


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))


def RMSLELoss(yhat, y):
    return torch.sqrt(torch.mean((torch.log(yhat + 1) - torch.log(y + 1)) ** 2))


def MSLELoss(yhat, y):
    return torch.mean((torch.log(yhat + 1) - torch.log(y + 1)) ** 2)


loss_dict = {
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
    "MSELoss": nn.MSELoss(),
    "PearsonLoss": PearsonLoss(),
    "RMSELoss": RMSELoss,
    "RMSLELoss": RMSLELoss,
    "MSLELoss": MSLELoss,
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
        elif self.hparams.model_name == "MDropModel":
            self.model = MDropModel(self.hparams.base_model_name, self.hparams.model_hparams)
        elif self.hparams.model_name == "WKPoolingModel":
            self.model = WKPoolingModel(self.hparams.base_model_name, self.hparams.model_hparams)
        elif self.hparams.model_name == "MeanPoolingModel":
            self.model = MeanPoolingModel(self.hparams.base_model_name, self.hparams.model_hparams)
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
        return {"preds": preds.squeeze(dim=-1), "labels": batch[1].squeeze(dim=-1), "val_loss": loss}

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

    df = pd.read_csv(config["input_dir"] + "folds.csv")
    cpc_texts = get_cpc_texts(config)
    df["context_text"] = df["context"].map(cpc_texts)
    if config["context_text_lower"]:
        df["text"] = df["anchor"] + "[SEP]" + df["target"] + "[SEP]" + df["context_text"].apply(str.lower)
    else:
        df["text"] = df["anchor"] + "[SEP]" + df["target"] + "[SEP]" + df["context_text"]

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

            filename = f"model-f{fold}-{{val_loss:.4f}}-{{val_score:.4f}}"
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
