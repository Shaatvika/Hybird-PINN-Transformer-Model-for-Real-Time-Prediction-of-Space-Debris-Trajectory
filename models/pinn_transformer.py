import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LayerNormalization
from models.orbital_physics import PhysicsLayer


class TransformerEncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.input_proj = Dense(d_model)  # Project input to d_model
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training=False):
        # Project input to d_model
        x = self.input_proj(x)
        
        x_expanded = tf.expand_dims(x, 1) if len(x.shape) == 2 else x
        attn = self.mha(x_expanded, x_expanded)
        attn = tf.squeeze(attn, 1) if len(x.shape) == 2 else attn

        out1 = self.norm1(x + self.dropout1(attn, training=training))
        ffn_out = self.ffn(out1)
        return self.norm2(out1 + self.dropout2(ffn_out, training=training))

class PINNTransformer(Model):
    def __init__(self, input_dim, output_dim, d_model=64, num_heads=4, dff=128, dropout_rate=0.1, **kwargs):
        super(PINNTransformer, self).__init__(**kwargs)
        self.physics = PhysicsLayer()  # Physics-informed layer
        self.transformer = TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)
        self.norm = LayerNormalization()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.output_layer = Dense(output_dim)

    def call(self, inputs, training=False):
        # Step 1: Apply physics layer
        physics_out = self.physics(inputs)

        # Step 2: Combine raw inputs with physics-informed features
        x = tf.concat([inputs, physics_out], axis=1)  # Shape (batch_size, 17)

        # Step 3: Apply transformer encoder
        x = self.transformer(x, training=training)

        # Step 4: Dense layers for final mapping
        x = self.norm(x)
        x = self.dense1(x)
        x = self.dense2(x)

        # Step 5: Output layer
        return self.output_layer(x)
