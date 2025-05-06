import tensorflow as tf
from models.pinn_transformer import PINNTransformer
import numpy as np

# Load the test data (ensure these paths match your saved data)
X_test = np.load('datasets/X_test.npy')  # Load the test data
y_test = np.load('datasets/y_test.npy')  # You can use this if you need to compare predictions

# Define the input and output dimensions (based on your test data)
input_dim = X_test.shape[1]  # Number of features in your input data
output_dim = y_test.shape[1]  # Number of target variables (output)

# Recreate the model with the same input and output dimensions
model = PINNTransformer(input_dim=input_dim, output_dim=output_dim)

# Build the model by calling it on some dummy input (X_test)
model.build((None, input_dim))  # (None, input_dim) is the shape of a batch with input_dim features

# Now load the saved model weights
model.load_weights('models/debris_predictor.keras')

# Now you can make predictions
predictions = model.predict(X_test)

# Example of using predictions (e.g., print them or save to a file)
print(predictions)

# Optionally, save the predictions as a numpy file for later use
np.save('predictions.npy', predictions)
