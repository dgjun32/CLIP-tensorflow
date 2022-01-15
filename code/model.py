import os
from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers


# Dot product attention layer
class DotProductAttention(layers.Layer):
    def __init__(self):
        super(DotProductAttention, self).__init__()
    
    def masked_fill_(self, tensor, mask):
        arr = tensor.numpy()
        arr[tf.where(mask == 0)] = float("-inf")
        return tf.Variable(arr)

    def build(self, input_shape):
        dim_k = tf.cast(input_shape[-1], tf.float32)
        self.scale = 1 / tf.sqrt(dim_k)
    
    def call(self, query, key, value, attn_mask):
        score = tf.matmul(query, key, transpose_b=True)
        if attn_mask is not None:
            score = self.masked_fill(score, attn_mask)
        score = tf.nn.softmax(score * self.scale)
        return tf.matmul(score, value)


# Multihead attention layer
class MultiheadAttention(layers.Layer):
    def __init__(self, num_heads, h_dim):
        super(MultiheadAttention, self).__init__()
        '''
        num_heads : # of attention head
        h_dim : dimension of hidden state
        '''
        self.num_heads = num_heads
        self.h_dim = h_dim
        units = h_dim // num_heads
        self.q_layers, self.k_layers, self.v_layers = [], [], []
        # multihead projection layers
        for _ in range(self.num_heads):
            layer = layers.Dense(units, activation=None, use_bias=False)
            self.q_layers.append(layer)
        for _ in range(self.num_heads):
            layer = layers.Dense(units, activation=None, use_bias=False)
            self.k_layers.append(layer)
        for _ in range(self.num_heads):
            layer = layers.Dense(units, activation=None, use_bias=False)
            self.v_layers.append(layer)
        # dot product attention layer
        self.attention = DotProductAttention()
        self.out = layers.Dense(self.h_dim, activation=None, use_bias=False)

    
    def call(self, query, key, value, attn_mask=None):
        q, k, v = query, key, value
        # linear projection for each head
        query = [l(q) for l in self.q_layers]
        key = [l(k) for l in self.k_layers]
        value = [l(v) for l in self.v_layers]
        # computing attention for each head
        head = [self.attention(q, k, v, attn_mask) for q, k, v in zip(query, key, value)]
        # concat output of each head -> linear projection
        out = self.out(tf.concat(head, -1))
        return out

# self attention block
class SelfAttentionLayer(layers.Layer):
    def __init__(self, num_heads, h_dim, m_dim, type = 'text', dropout_rate = 0.1):
        super(SelfAttentionLayer, self).__init__()
        self.type = type
        self.mha = MultiheadAttention(num_heads=num_heads, h_dim=h_dim)
        self.skip = layers.Add()
        self.layernorm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([layers.Dense(m_dim, activation=tf.nn.gelu),
                                        layers.Dense(h_dim, activation=None)])
    def call(self, x, attn_mask=None):
        x1 = self.layernorm_1(x)
        x2 = self.mha(query=x1, value=x1, key=x1, attn_mask = attn_mask)
        x = self.skip([x2, x])
        x1 = self.layernorm_2(x)
        x2 = self.mlp(x1)
        x = self.skip([x2, x])
        return x

############### Vision Transformer ######################

# image to patches
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
            )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# image patches to projection
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
            )
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

# vision transformer
class VisionTransformer(layers.Layer):
    def __init__(self, num_heads, n_layers, h_dim, m_dim, patch_size, img_size = 224):
        super(VisionTransformer, self).__init__()
        # hyperparams of vision transformer
        num_patches = (img_size // patch_size)**2
        self.patches = Patches(patch_size)
        self.pre_proj = PatchEncoder(num_patches, h_dim)
        self.pre_ln = layers.LayerNormalization(axis=1)
        self.encoder = tf.keras.Sequential([SelfAttentionLayer(num_heads, h_dim, m_dim, type='image')
                                                for _ in range(n_layers)])
        self.post_ln = layers.LayerNormalization(axis=1)
        self.post_proj = layers.Dense(h_dim, activation='linear')
    def call(self, x):
        x = self.patches(x)
        x = self.pre_proj(x)
        x = self.pre_ln(x)
        x = self.encoder(x)
        x = self.post_ln(x)
        x = self.post_proj(x)
        return x

######################### text transformer #############################

# tokens to h_dim dimensional embedding
class Embedding(layers.Layer):
    def __init__(self, vocab_size, context_size, h_dim):
        super(Embedding, self).__init__()
        self.embedding = layers.Embedding(vocab_size, h_dim)
        self.pos_encoding = tf.Variable(np.random.randn(context_size, h_dim), dtype=tf.float32)
    def call(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoding
        return x

class TextTransformer(layers.Layer):
    def __init__(self, num_heads, n_layers, vocab_size, context_size, h_dim, m_dim):
        super(TextTransformer, self).__init__()
        self.embedding = Embedding(vocab_size, context_size, h_dim)
        self.encoder = tf.keras.Sequential([SelfAttentionLayer(num_heads, h_dim, m_dim, type='text')
                                            for _ in range(n_layers)])
        self.post_ln = layers.LayerNormalization(axis=1)
    def call(self, x, attn_mask):
        x = self.embedding(x)
        x = self.encoder(x, attn_mask)
        x = self.post_ln(x)
        return x


################################ CLIP ########################################
class CLIPModel(tf.keras.Model):
    def __init__(self, name):
        '''
        name : [Str] One of 'vit-B/32' ,'vit-B/16', 'vit-L/14'
        '''
        super(CLIPModel, self).__init__()
        # image encoder
        if name == 'vit-B/32':
            self.image_encoder = VisionTransformer(num_heads=12, n_layers=12, h_dim=768, m_dim=3072, patch_size=32)
        elif name == 'vit-B/16':
            self.image_encoder = VisionTransformer(num_heads=12, n_layers=12, h_dim=768, m_dim=3072, patch_size=16)
        elif name == 'vit-L/14':
            self.image_encoder = VisionTransformer(num_heads=16, n_layers=24, h_dim=1024, m_dim=4096, patch_size=14)
        # text encoder
        self.text_encoder = TextTransformer(num_heads=12, n_layers=12, vocab_size=30000, context_size=76, h_dim=512, m_dim=2048)
        self.image_proj = layers.Dense(512, activation='linear')
        self.text_proj = layers.Dense(512, activation='linear')
    
    def encode_text(self, tokens, attn_mask):
        out = self.text_encoder(tokens, attn_mask)
        return out
    
    def encode_image(self, image):
        out = self.image_encoder(image)
        return out

    def call(self, image, text, attn_mask):
        '''
        image : torch.FloatTensor of shape (batch_size, 3, 224, 224)
        text : Dictionary of torch.LongTensor, torch.BoolTensor of shape (batch_size, 77)
        attn_mask : 
        '''
        image_features = self.image_proj(self.encode_image(image))
        print(image_features.shape)
        text_features = self.text_proj(self.encode_text(text, attn_mask))
        
        # consine similarity
        image_features = tf.keras.utils.normalize(image_features, axis=-1)[:,0,:]
        text_features = tf.keras.utils.normalize(text_features, axis=-1)[:,0,:]
        image_logits = tf.matmul(image_features, tf.transpose(text_features))
        text_logits = tf.transpose(image_logits)

        return image_logits, text_logits