import time
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import overall_performance
import numpy as np
import random


def train(epoch, model, loss_fn_md, loss_fn_wsd, loss_fn_task, optimizer, train_loader, total_steps, LamdaSetter, args, scheduler=None):
    epoch_start_time = time.time()
    model.train()
    tr_wsd_loss = 0  # training loss in current epoch
    tr_md_loss = 0
    tr_global_loss = 0
    tr_local_loss = 0
    # ! training
    for step, batch in enumerate(tqdm(train_loader, desc='Iteration')):
        # unpack batch data
        batch = tuple(t.cuda() for t in batch)
        input_ids, type_ids, att_mask, labels, domains = batch  # domains: [1 x n]
        adv_lmbd = LamdaSetter.set_lambda_(step + epoch * len(train_loader), total_steps)
        half_batch_size = int(0.5 * len(labels))

        # compute logits
        wsd_out, md_out, global_out, wsd_local_out, md_local_out = model(input_ids, type_ids, att_mask, adv_lmbd)

        wsd_labels = labels[:half_batch_size]
        md_labels = labels[half_batch_size:]
        wsd_domains = domains[:half_batch_size]
        md_domains = domains[half_batch_size:]

        # compute loss
        wsd_loss = loss_fn_wsd(wsd_out, wsd_labels)
        wsd_local_loss0 = loss_fn_task(wsd_local_out[0], wsd_domains)
        wsd_local_loss1 = loss_fn_task(wsd_local_out[1], wsd_domains)

        md_loss = loss_fn_md(md_out, md_labels)
        global_loss = loss_fn_task(global_out, domains)

        md_local_loss0 = loss_fn_task(md_local_out[0], md_domains)
        md_local_loss1 = loss_fn_task(md_local_out[1], md_domains)

        local_loss = 0.3 * wsd_local_loss0 + 0.3 * wsd_local_loss1 + md_local_loss0 + md_local_loss1

        tr_wsd_loss += wsd_loss.item()
        tr_md_loss += md_loss.item()
        tr_global_loss += global_loss.item()
        tr_local_loss += local_loss.item()

        loss = md_loss + args.TRAIN.wsd_weight * wsd_loss + args.TRAIN.global_weight * global_loss + args.TRAIN.local_weight * local_loss

        # back propagation
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()  # adjusting learning rate
        optimizer.zero_grad()
        step += 1

    timing = time.time() - epoch_start_time
    cur_lr = optimizer.param_groups[0]["lr"]
    print(f"Timing: {timing}, Epoch: {epoch + 1}\n"
          f"md loss: {tr_md_loss}, "
          f"wsd loss: {tr_wsd_loss}\n"
          f"global loss: {tr_global_loss}, "
          f"local loss: {tr_local_loss}, "
          f"current learning rate {cur_lr}")


def val(model, val_loader, domain_idx):
    # make sure to open the eval mode.
    model.eval()

    # prepare loss function
    loss_fn = nn.CrossEntropyLoss()

    val_loss = 0
    val_preds = []
    val_labels = []
    for batch in val_loader:
        # unpack batch data
        batch = tuple(t.cuda() for t in batch)
        input_ids, type_ids, att_mask, labels, domains = batch

        with torch.no_grad():
            # compute logits
            out = model(input_ids, type_ids, att_mask, domain_idx=domain_idx)
            # get the prediction labels
            preds = torch.max(out.data, 1)[1].cpu().numpy().tolist()  # prediction labels [1, batch_size]
            # compute loss
            loss = loss_fn(out, labels)
            val_loss += loss.item()

            labels = labels.cpu().numpy().tolist()  # ground truth labels [1, batch_size]
            val_labels.extend(labels)
            val_preds.extend(preds)
    print(f"val loss: {val_loss}")

    # get overall performance
    val_acc, val_prec, val_recall, val_f1 = overall_performance(val_labels, val_preds)
    return val_acc, val_prec, val_recall, val_f1


def set_random_seeds(seed):
    """
    set random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
