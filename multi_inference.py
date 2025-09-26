import tensorflow as tf
import numpy as np
from models.pinn_transformer import PINNTransformer

# Load test data
X_test = np.load('datasets/X_test.npy')
y_test = np.load('datasets/y_test.npy')  # Optional

# Model dimensions
input_dim = X_test.shape[1]
output_dim = y_test.shape[1]

# Recreate model architecture
model = PINNTransformer(input_dim=input_dim, output_dim=output_dim)
model.build((None, input_dim))

# Load saved weights
model.load_weights('models/debris_predictor_weights')

# Multi-step prediction
n_steps = 1000
initial_input = X_test[0]
multi_step_preds = []

current_input = initial_input
for _ in range(n_steps):
    next_pred = model.predict(np.expand_dims(current_input, axis=0))[0]
    multi_step_preds.append(next_pred)
    current_input = next_pred  # Feed prediction as next input

multi_step_preds = np.array(multi_step_preds)

# Save results
np.save("multi_step_predictions.npy", multi_step_preds)
print(f"Saved {n_steps} steps of predictions to multi_step_predictions.npy")
