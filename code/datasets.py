import os 
import pathlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf


# load normalized images and text caption
class PairDataLoader(tf.keras.utils.Sequence):
    def __init__(self, cfg):
        super(PairDataLoader, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size
        with open(cfg.path.train_annot) as f:
            self.train_annot = json.load(f)
        self._to_dataframe()
        self.indices = np.arange(len(self.train_df))
        np.random.shuffle(self.indices)

    def _to_dataframe(self):
        # for train dataset
        images = self.train_annot['images']
        annots = self.train_annot['annotations']
        image_ids, image_paths, annot_ids, annot_cap = [], [], [], []
        for sample in images:
            image_ids.append(sample['id'])
            image_paths.append(os.path.join(self.cfg.path.train_image, sample['file_name']))
        for sample in annots:
            annot_ids.append(sample['image_id'])
            annot_cap.append(sample['caption'])
        train_a = pd.DataFrame({'id':image_ids, 'image_path':image_paths})
        train_b = pd.DataFrame({'id':annot_ids, 'caption':annot_cap})
        self.train_df = train_a.merge(train_b, on='id', how='left')
    
    def _img_preprocess(self, img_path):
        proc = tf.keras.Sequential([
            tf.keras.layers.Resizing(224, 224),
            tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)
        ])
        img = tf.keras.utils.load_img(img_path)
        img = tf.keras.utils.img_to_array(img)
        img = tf.constant(img)
        img_tensor = proc(img)
        return img_tensor

    def __len__(self):
        return int(len(self.train_df)//self.batch_size)

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        images = np.array([self._img_preprocess(self.train_df.image_path[i]) for i in indices])
        captions = [self.train_df.caption[i] for i in indices]

        return tf.convert_to_tensor(images, dtype=tf.float32), captions

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        
# tokenize caption into tokens and attention_mask
def prepare_data(batch, tokenizer):
    pixel, caption = batch
    tokens = tokenizer.batch_encode_plus(caption, max_length=76, padding='max_length', return_tensors='tf')
    attn_mask = tokens['attention_mask']
    attn_mask = tf.matmul(tf.expand_dims(attn_mask, 2), tf.transpose(tf.expand_dims(attn_mask, 2), perm=(0,2,1)))
    return pixel, tokens['input_ids'], attn_mask
        

class ImageDataLoader(tf.keras.utils.Sequence):
    def __init__(self, batch_size, img_dir, annot_dir):
        '''
        img_dir : directory containing images
        annot_dir : .json file path containing annotation
        '''
        super(ImageDataLoader, self).__init__()
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        with open(annot_dir) as f:
            self.annot = json.load(f)
        self._to_dataframe()
        self.indices = np.arange(len(self.df))

    def _to_dataframe(self):
        # for train dataset
        images = self.annot['images']
        image_ids, image_paths = [], []
        for sample in images:
            image_ids.append(sample['id'])
            image_paths.append(os.path.join(self.img_dir, sample['file_name']))
        self.df = pd.DataFrame({'id':image_ids, 'image_path':image_paths})
    
    def return_df(self):
        return self.df
    
    def _img_preprocess(self, img_path):
        proc = tf.keras.Sequential([
            tf.keras.layers.Resizing(224, 224),
            tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)
        ])
        img = tf.keras.utils.load_img(img_path)
        img = tf.keras.utils.img_to_array(img)
        img = tf.constant(img)
        img_tensor = proc(img)
        return img_tensor

    def __len__(self):
        return int(len(self.df)//self.batch_size)

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        images = np.array([self._img_preprocess(self.df.image_path[i]) for i in indices])
        return tf.convert_to_tensor(images, dtype=tf.float32)

