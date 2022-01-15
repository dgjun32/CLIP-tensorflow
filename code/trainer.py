import os
import numpy as np
import tensorflow as tf

from config import cfg
from datasets import DataLoader, prepare_data


class Trainer:
    def __init__(self, cfg, model, tokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = tf.Variable(0.07, trainable=True)
    def train(self):
        # set dataloader & optimizer 
        dataloader = DataLoader(self.cfg)
        optimizer = tf.keras.optimizers.Adam()
        for epoch in range(self.cfg.train.num_epochs):
            print('-'*50)
            print('Epoch {}'.format(epoch+1))
            epoch_loss = 0.0
            for step, batch in enumerate(dataloader):
                # preprocessing data
                pixel, input_ids, attn_mask = prepare_data(batch, self.tokenizer)
                with tf.GradientTape() as tape:
                    # forward propagate
                    image_logits, text_logits = self.model(pixel, input_ids, attn_mask)
                    image_logits = image_logits * self.temperature
                    text_logits = text_logits * self.temperature
                    label = tf.eye(cfg.train.batch_size)
                    loss_i = tf.nn.softmax_cross_entropy_with_logits(label, image_logits)
                    loss_t = tf.nn.softmax_cross_entropy_with_logits(label, text_logits)
                    loss = tf.math.reduce_sum(loss_i)/2 + tf.math.reduce_sum(loss_t)/2
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                epoch_loss += loss.numpy()

                # verbosity
                if (step+1) % 100 == 0:
                    print('| step {} | train_loss : {} |'.format(step+1, epoch_loss/step))
        print("Training Complete")
        self.model.save_weights('./output/clip_32epoch_trained_with_MScoco.ckpt')
                



            




    
    