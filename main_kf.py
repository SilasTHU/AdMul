import os
import os.path as osp
import random
import sys
import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AdamW
import torch.nn as nn
import time

from model import DeBERTa_base
from data_utils import dataset, collate_fn, BalanceSampler
from data_process import WSD_Processor, MD_Processor
from utils import Logger, Lamda
from configs.default import get_config
from train_val import train, val, set_random_seeds


def parse_option():
    parser = argparse.ArgumentParser(description='Train on MOH-X dataset, do cross validation')
    parser.add_argument('--cfg', type=str, default='./configs/mohx.yaml', metavar="FILE",
                        help='path to config file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids')
    parser.add_argument('--seed', default=4, type=int, help='random seed')
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--task', default='mohx', type=str, help='train on vua verb or vua all')
    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config


def get_kfold_data(k, i, raw_data):
    fold_size = len(raw_data) // k

    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        val_raw = raw_data[val_start:val_end]
        tr_raw = raw_data[0:val_start] + raw_data[val_end:]
    else:
        val_raw = raw_data[val_start:]
        tr_raw = raw_data[0:val_start]

    return tr_raw, val_raw


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sys.stdout = Logger(osp.join(args.TRAIN.output,
                                 f'{args.task}_{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())}.txt'))
    print(args)
    set_random_seeds(args.seed)

    wp = WSD_Processor(args)
    mp = MD_Processor(args)

    if args.task == 'mohx':
        md_data = mp.get_mohx()
    else:
        md_data = mp.get_trofi()
    random.shuffle(md_data)
    wsd_train_data = wp.get_wsd_train()
    random.shuffle(wsd_train_data)
    wsd_test_data = wp.get_wsd_test()

    accs = 0
    pres = 0
    recs = 0
    f1s = 0
    for i in range(10):
        print('*' * 20, f"training on fold #{i + 1}", '*' * 20)
        md_train_data, md_test_data = get_kfold_data(10, i, md_data)
        md_train_set = dataset(md_train_data)
        md_test_set = dataset(md_test_data)

        if len(wsd_train_data) > len(md_train_data):
            wsd_train_data = wsd_train_data[:len(md_train_data)]
        wsd_train_set = dataset(wsd_train_data)
        wsd_test_set = dataset(wsd_test_data)

        train_set = ConcatDataset([wsd_train_set, md_train_set])

        train_loader = DataLoader(dataset=train_set,
                                  sampler=BalanceSampler(dataset=train_set, batch_size=args.TRAIN.train_batch_size),
                                  batch_size=args.TRAIN.train_batch_size,
                                  collate_fn=collate_fn,
                                  shuffle=False)

        wsd_test_loader = DataLoader(dataset=wsd_test_set,
                                     batch_size=args.TRAIN.val_batch_size,
                                     collate_fn=collate_fn,
                                     shuffle=False)

        md_test_loader = DataLoader(dataset=md_test_set,
                                    batch_size=args.TRAIN.val_batch_size,
                                    collate_fn=collate_fn,
                                    shuffle=False)
        # load model
        model = DeBERTa_base(args)
        model.cuda()

        # prepare optimizer
        optimizer = AdamW(model.parameters(), lr=args.TRAIN.lr, weight_decay=args.TRAIN.weight_decay)

        # prepare loss function
        loss_fn_wsd = nn.CrossEntropyLoss()
        loss_fn_md = nn.CrossEntropyLoss(weight=torch.Tensor([1, args.TRAIN.md_class_weight]).cuda())
        loss_fn_task = nn.CrossEntropyLoss(weight=torch.Tensor([1, args.TRAIN.wsd_class_weight]).cuda())

        LamdaSetter = Lamda(lo=args.TRAIN.lambda_lo, hi=args.TRAIN.lambda_hi)
        total_steps = len(train_loader) * args.TRAIN.train_epochs

        best_acc = -1
        best_pre = -1
        best_rec = -1
        best_f1 = -1
        for epoch in range(args.TRAIN.train_epochs):
            print('===== Start training: epoch {} ====='.format(epoch + 1))
            train(epoch, model, loss_fn_md, loss_fn_wsd, loss_fn_task, optimizer, train_loader, total_steps, LamdaSetter, args)
            t_a, t_p, t_r, t_f1 = val(model, md_test_loader, domain_idx=1)
            val(model, wsd_test_loader, domain_idx=0)
            if t_f1 > best_f1:
                best_acc, best_pre, best_rec, best_f1 = t_a, t_p, t_r, t_f1
        accs += best_acc
        pres += best_pre
        recs += best_rec
        f1s += best_f1

    print('average result:')
    print(accs / 10)
    print(pres / 10)
    print(recs / 10)
    print(f1s / 10)


if __name__ == '__main__':
    args = parse_option()
    main(args)
