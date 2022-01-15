import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers

from trainer import Trainer 
from model import CLIPModel
from config import cfg



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CLIP')
    parser.add_argument('--model_name', dest='name', type=str, default='vit-B/32')
    args = parser.parse_args()

    tokenizer = transformers.CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel(args.name)
    algo = Trainer(cfg, model, tokenizer)
    algo.train()
