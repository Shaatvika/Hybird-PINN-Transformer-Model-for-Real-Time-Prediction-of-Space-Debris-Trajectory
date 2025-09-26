import tensorflow as tf
import numpy as np
from models.pinn_transformer import PINNTransformer

# Load test data
X_test = np.load('datasets/X_test.npy')
y_test = np.load('datasets/y_test.npy')  # Optional, for comparison

# Model dimensions
input_dim = X_test.shape[1]
output_dim = y_test.shape[1]

# Recreate model architecture
model = PINNTransformer(input_dim=input_dim, output_dim=output_dim)
model.build((None, input_dim))

# Load saved weights
model.load_weights('models/debris_predictor_weights')

# Make predictions
predictions = model.predict(X_test)

# Print and save predictions
print(predictions)
np.save('predictions.npy', predictions)
