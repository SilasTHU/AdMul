from yacs.config import CfgNode as CN

_C = CN()

"""
    data settings
"""
_C.DATA = CN()
_C.DATA.wsd_threshold = 3
_C.DATA.data_dir = './data'
_C.DATA.max_len = 150
_C.DATA.plm = './deberta-v3-base'
"""
    model settings
"""
_C.MODEL = CN()
_C.MODEL.num_classes = 2
_C.MODEL.num_tasks = 2
# embedding dim
_C.MODEL.embed_dim = 768
# drop out rate
_C.MODEL.dropout = 0.2

'''
    training settings
'''
_C.TRAIN = CN()
_C.TRAIN.train_batch_size = 32
_C.TRAIN.val_batch_size = 32
_C.TRAIN.lr = 3e-5
_C.TRAIN.train_epochs = 5
_C.TRAIN.warmup_epochs = 2
# the directory to save the training logs
_C.TRAIN.output = './data/logs'
_C.TRAIN.md_class_weight = 4
_C.TRAIN.wsd_class_weight = 4
_C.TRAIN.weight_decay = 0.01
_C.TRAIN.wsd_weight = 0.2
_C.TRAIN.local_weight = 0.1
_C.TRAIN.global_weight = 0.1
_C.TRAIN.lambda_lo = 0.1
_C.TRAIN.lambda_hi = 0.8
_C.gpu = '0'
_C.seed = 4
# do eval only
_C.eval_mode = False
_C.log = 'log_test'
_C.task = 'verb'


def update_config(config, args):
    config.defrost()

    print('=> merge config from {}'.format(args.cfg))
    config.merge_from_file(args.cfg)

    if args.gpu:
        config.gpu = args.gpu

    if args.seed:
        config.seed = args.seed

    if args.eval:
        config.eval_mode = True

    if args.task:
        config.task = args.task
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)

    return config
