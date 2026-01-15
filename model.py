import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CausalSelfAttention(layers.Layer):
    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        self.qkv = layers.Dense(3 * d_model, use_bias=False)
        self.proj = layers.Dense(d_model, use_bias=False)
        self.attn_drop = layers.Dropout(dropout)
        self.resid_drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        return