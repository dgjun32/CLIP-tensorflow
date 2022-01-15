import os
from easydict import EasyDict as edict

cfg = edict()

cfg.path = edict()
cfg.path.train_image = "../data/train2017"
cfg.path.val_image = "../data/val2017"
cfg.path.test_image = "../data/test2017"
cfg.path.train_annot = "../data/annotations/captions_train2017.json"
cfg.path.val_annot = "../data/annotations/captions_val2017.json"

cfg.train = edict()
cfg.train.img_size = 224
cfg.train.num_epochs = 32
cfg.train.batch_size = 32678





