import torch
import time
import numpy as np
from .meter import AverageMeter
from .utils import timeSince


def valid_fn(valid_loader, model, criterion, device, CFG):
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
