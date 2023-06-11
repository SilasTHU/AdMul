import os
import os.path as osp
import sys
import argparse
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch
import time
import random

from model import DeBERTa_base
from data_utils import dataset, collate_fn, BalanceSampler
from data_process import WSD_Processor, MD_Processor
from utils import Logger, Lamda
from configs.default import get_config
from train_val import train, val, set_random_seeds


def parse_option():
    parser = argparse.ArgumentParser(description='Train on VUA dataset')
    parser.add_argument('--cfg', type=str, default='./configs/vua_verb.yaml', metavar="FILE",
                        help='path to config file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids')
    parser.add_argument('--seed', default=4, type=int, help='random seed')
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--task', default='verb', type=str, help='train on vua verb or vua all')
    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sys.stdout = Logger(osp.join(args.TRAIN.output,
                                 f'{args.task}_{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())}.txt'))
    print(args)
    set_random_seeds(args.seed)

    wp = WSD_Processor(args)
    mp = MD_Processor(args)

    if args.eval_mode:
        print("Evaluate only")
        if args.task == 'all':
            model = DeBERTa_base(args)
            model.cuda()
            model.load_state_dict(torch.load('./best_check.pth'))
            test_data = mp.get_all_test()
            acad_data = mp.get_acad()
            conv_data = mp.get_conv()
            fict_data = mp.get_fict()
            news_data = mp.get_news()
            adj_data = mp.get_adj()
            adv_data = mp.get_adv()
            noun_data = mp.get_noun()
            verb_data = mp.get_verb()

            test_set = dataset(test_data)
            acad_set = dataset(acad_data)
            conv_set = dataset(conv_data)
            fict_set = dataset(fict_data)
            news_set = dataset(news_data)
            adj_set = dataset(adj_data)
            adv_set = dataset(adv_data)
            noun_set = dataset(noun_data)
            verb_set = dataset(verb_data)

            test_loader = DataLoader(test_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
            acad_loader = DataLoader(acad_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
            conv_loader = DataLoader(conv_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
            fict_loader = DataLoader(fict_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
            news_loader = DataLoader(news_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
            adj_loader = DataLoader(adj_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
            adv_loader = DataLoader(adv_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
            noun_loader = DataLoader(noun_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
            verb_loader = DataLoader(verb_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)

            # transfer learning part
            trofi_data = mp.get_trofi()
            trofi_set = dataset(trofi_data)
            trofi_loader = DataLoader(trofi_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
            mohx_data = mp.get_mohx()
            mohx_set = dataset(mohx_data)
            mohx_loader = DataLoader(mohx_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)

            print('-------test-------')
            val(model, test_loader, domain_idx=1)
            print('-------acad-------')
            val(model, acad_loader, domain_idx=1)
            print('-------conv-------')
            val(model, conv_loader, domain_idx=1)
            print('-------fict-------')
            val(model, fict_loader, domain_idx=1)
            print('-------news-------')
            val(model, news_loader, domain_idx=1)
            print('-------adj-------')
            val(model, adj_loader, domain_idx=1)
            print('-------adv-------')
            val(model, adv_loader, domain_idx=1)
            print('-------noun-------')
            val(model, noun_loader, domain_idx=1)
            print('-------verb-------')
            val(model, verb_loader, domain_idx=1)
            print('-------trofi-------')
            val(model, trofi_loader, domain_idx=1)
            print('-------mohx-------')
            val(model, mohx_loader, domain_idx=1)
            return
        if args.task == 'verb':
            model = DeBERTa_base(args)
            model.cuda()
            model.load_state_dict(torch.load('./best_check.pth'))
            test_data = mp.get_verb_test()
            test_set = dataset(test_data)
            test_loader = DataLoader(test_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
            print('-------test-------')
            val(model, test_loader, domain_idx=1)

            return

    if args.task == 'verb':
        md_train_data = mp.get_verb_train()
        md_test_data = mp.get_verb_test()
        md_val_data = mp.get_verb_val()
    else:
        md_train_data = mp.get_all_train()
        md_val_data = mp.get_all_val()
        md_test_data = mp.get_all_test()

    md_train_set = dataset(md_train_data)
    md_val_set = dataset(md_val_data)
    md_test_set = dataset(md_test_data)

    wsd_train_data = wp.get_wsd_train()
    if len(wsd_train_data) >= len(md_train_data):
        wsd_train_data = wsd_train_data[:len(md_train_data)]
    random.shuffle(wsd_train_data)
    wsd_test_data = wp.get_wsd_test()

    wsd_train_set = dataset(wsd_train_data)
    wsd_test_set = dataset(wsd_test_data)

    # concat wsd dataset and metaphor detection dataset
    # make sure to put the smaller dataset as the first element
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
    md_val_loader = DataLoader(dataset=md_val_set,
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
    if args.task == 'all':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.TRAIN.warmup_epochs * len(train_loader)),
            num_training_steps=total_steps,
        )
    else:
        scheduler = None
    best_f1 = 0
    for epoch in range(args.TRAIN.train_epochs):
        print('===== Start training: epoch {} ====='.format(epoch + 1))
        train(epoch, model, loss_fn_md, loss_fn_wsd, loss_fn_task, optimizer, train_loader, total_steps, LamdaSetter, args, scheduler=scheduler)
        a, p, r, f1 = val(model, md_val_loader, domain_idx=1)
        val(model, md_test_loader, domain_idx=1)
        val(model, wsd_test_loader, domain_idx=0)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), './best_check.pth')


if __name__ == '__main__':
    args = parse_option()
    main(args)
