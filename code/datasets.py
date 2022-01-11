import os 
import pathlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

from config import cfg


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, cfg):
        super(DataLoader, self).__init__()
        with open(cfg.path.train_annot) as f:
            train_annot = json.laod(f)
        with open(cfg.path.val_annot) as f:
            val_annot = json.load(f)
        self.train_df = pd.DataFrame()
        images = train_annot['images']
        annots = train_annot['annotations']
        
    def 
