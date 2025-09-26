import os
import numpy as np
import tensorflow as tf
from utils.data_processing import load_and_preprocess_data
from models.pinn_transformer import PINNTransformer
from utils.config import Config
import matplotlib.pyplot as plt

# Enable eager execution for debugging
tf.config.run_functions_eagerly(True)

# Disable GPU if not available
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Ensure required folders exist
os.makedirs("datasets", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load config
config = Config(batch_size=32, epochs=100, learning_rate=1e-4, model_path='models/debris_predictor_weights')

# Function to plot training history
def plot_training(history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.yscale('log')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['root_mean_squared_error'], label='RMSE')
    plt.plot(history.history['val_root_mean_squared_error'], label='Val RMSE')
    plt.yscale('log')
    plt.legend()
    plt.savefig("training_history.png")
    plt.show()

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, _, _ = load_and_preprocess_data('data/raw/tle_data.csv')

    # Save test data for inference
    np.save("datasets/X_test.npy", X_test)
    np.save("datasets/y_test.npy", y_test)

    # Prepare TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(config.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(config.batch_size)

    # Create model
    model = PINNTransformer(input_dim=X_train.shape[1], output_dim=y_train.shape[1])

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

    # Train model with callbacks
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=config.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(filepath=config.model_path, save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
    )

    # Save only weights (safe for subclassed models)
    model.save_weights(config.model_path)
    print(f"Model weights saved to {config.model_path}")

    # Plot training history
    plot_training(history)

if __name__ == '__main__':
    main()
