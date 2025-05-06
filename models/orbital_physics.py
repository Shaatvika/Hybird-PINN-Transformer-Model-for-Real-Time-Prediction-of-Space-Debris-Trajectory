import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class PhysicsLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mu = 3.986004418e14  # Gravitational parameter (m^3/s^2)

    def call(self, inputs):
        mean_motion = inputs[:, 1] * 0.1 + 0.05
        eccentricity = inputs[:, 2] * 0.5
        inclination = inputs[:, 3] * np.pi
        raan = inputs[:, 4] * 2 * np.pi
        arg_perigee = inputs[:, 5] * 2 * np.pi
        mean_anomaly = inputs[:, 6] * 2 * np.pi

        n = mean_motion * (2 * np.pi) / 60.0
        a = (self.mu / (n**2)) ** (1/3)

        E = mean_anomaly
        for _ in range(10):
            E = mean_anomaly + eccentricity * tf.sin(E)

        x = a * (tf.cos(E) - eccentricity)
        y = a * tf.sqrt(1 - eccentricity**2) * tf.sin(E)
        z = tf.zeros_like(x)

        x_eci = x * tf.cos(raan) - y * tf.cos(inclination) * tf.sin(raan)
        y_eci = x * tf.sin(raan) + y * tf.cos(inclination) * tf.cos(raan)
        z_eci = y * tf.sin(inclination)

        return tf.stack([
            a / 1e7,
            tf.sqrt(x**2 + y**2 + z**2) / 1e7,
            x_eci / 1e7,
            y_eci / 1e7,
            z_eci / 1e7,
            n,
            eccentricity
        ], axis=1)
