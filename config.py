from yacs.config import CfgNode as CN

_C = CN()

# Model
_C.MODEL = CN()
_C.MODEL.ARCH = "se_resnext50_32x4d"  # check python train.py -h for available models
_C.MODEL.IMG_SIZE = 224

# Train
_C.TRAIN = CN()
_C.TRAIN.OPT = "sgd"  # adam or sgd
_C.TRAIN.WORKERS = 8
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_DECAY_STEP = 20
_C.TRAIN.LR_DECAY_RATE = 0.2
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.EPOCHS = 60
_C.TRAIN.AGE_STDDEV = 1.0

# Test
_C.TEST = CN()
_C.TEST.WORKERS = 8
_C.TEST.BATCH_SIZE = 64
