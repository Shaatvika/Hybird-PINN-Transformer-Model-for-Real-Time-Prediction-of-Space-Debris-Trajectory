import tensorflow as tf
from models.orbital_physics import PhysicsLayer
from tensorflow.keras import layers
from keras import Model
from keras.layers import Dense, Input, LayerNormalization

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        x_expanded = tf.expand_dims(x, 1) if len(x.shape) == 2 else x
        attn = self.mha(x_expanded, x_expanded)
        attn = tf.squeeze(attn, 1) if len(x.shape) == 2 else attn
        out1 = self.norm1(x + self.dropout1(attn, training=training))
        ffn_out = self.ffn(out1)
        return self.norm2(out1 + self.dropout2(ffn_out, training=training))


class PINNTransformer(Model):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(PINNTransformer, self).__init__(**kwargs)
        self.norm = LayerNormalization()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.output_layer = Dense(output_dim)

    def call(self, inputs):
        x = self.norm(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)
