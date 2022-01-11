import os
import numpy as np
import tensorflow as tf

from config import cfg

def train(clip, cfg, optimizer, scheduler, loss_fn, batch, label):
    image_logits, text_logits = clip(**batch)

    image_logits = image_logits * tf.exp(cfg.train.temperature)
    text_logits = text_logits * tf.exp(cfg.train.temperature)
    loss_i = loss_fn(image_logits)
    loss_t = loss_fn(text_logits)
    
    