import os
from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers


# self attention block
class AttentionLayer(layers.Layer):
    def __init__(self, num_heads, h_dim, m_dim, dropout_rate = 0.1):
        super(AttentionLayer, self).__init__()
        self.mha = layers.MultiHeadAttention(num_heads, h_dim, dropout=0.1)
        self.skip = layers.Add()
        self.layernorm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([layers.Dense(m_dim, activation=tf.nn.gelu),
                                        layers.Dropout(dropout_rate),
                                        layers.Dense(h_dim, activation='linear'),
                                        layers.Dropout(dropout_rate)])
    def call(self, x):
        x1 = self.layernorm_1(x)
        x2 = self.mha(x1, x1)
        x = self.skip(x2, x)
        x1 = self.layernorm_2(x)
        x2 = self.mlp(x1)
        x = self.skip(x2, x)
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
        self.encoder = tf.keras.Sequential([AttentionLayer(num_heads, h_dim, m_dim)
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
        self.pos_encoding = tf.Variable(np.random.randn((context_size, h_dim)))
    def call(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoding
        return x

class TextTransformer(layers.Layer):
    def __init__(self, num_heads, n_layers, vocab_size, context_size, h_dim, m_dim):
        super(TextTransformer, self).__init__()
        self.embedding = Embedding(vocab_size, context_size, h_dim)
        self.encoder = tf.keras.Sequential([AttentionLayer(num_heads, h_dim, m_dim)
                                            for _ in range(n_layers)])
        self.post_ln = layers.LayerNormalization(axis=1)
    def call(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
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
        self.text_encoder = TextTransformer(num_heads=12, n_layers=12, vocab_size=30000, context_size=77, h_dim=512, m_dim=2048)
    
    def encode_text(self, tokens):
        out = self.text_encoder(tokens)
        return out
    
    def encode_image(self, image):
        out = self.image_encoder(image)
        return out

    def forward(self, image, text):
        img_features = self.encode_image(image)
        text_features = self.encode_text(text)

        img_features = tf.keras.utils.normalize(img_features, axis=-1)
        text_features = tf.keras.utils.normalize(text_features, axis=-1)

