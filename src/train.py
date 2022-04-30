import torch
import gc
import time
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from .meter import AverageMeter
from .utils import timeSince
from .data import TrainDataset
from .model import CustomModel
from .valid import valid_fn
from .utils import get_score


def train_loop(folds, fold, device, LOGGER, CFG):

    LOGGER.info(f"\n###### Fold {fold}")

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
        pin_memory=False,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=False,
        drop_last=False,
    )

    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, CFG["output_dir"] + "config.pth")
    model.to(device)

    optimizer_parameters = get_optimizer_params(
        model, encoder_lr=CFG["encoder_lr"], decoder_lr=CFG["decoder_lr"], weight_decay=CFG["weight_decay"]
    )
    optimizer = AdamW(
        optimizer_parameters, lr=CFG["encoder_lr"], eps=CFG["eps"], betas=(CFG["betas"][0], CFG["betas"][1])
    )

    num_train_steps = len(train_loader) * CFG["epochs"]
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    if CFG["loss"] == "BCE":
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
    else:
        criterion = nn.L1Loss(reduction="mean")

    best_score = 0.0

    for epoch in range(CFG["epochs"]):

        start_time = time.time()

        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, CFG)
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device, CFG)

        score = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time

        LOGGER.info(f"Epoch {epoch+1} - train_loss: {avg_loss:.4f}  val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s")
        LOGGER.info(f"Epoch {epoch+1} - Score: {score:.4f}")

        if best_score < score:
            best_score = score
            LOGGER.info(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f}")
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


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, CFG):
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
