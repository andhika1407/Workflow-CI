## import library
import os
import sys
import pandas as pd
import mlflow
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
  lstm_units = int(sys.argv[1]) if len(sys.argv) > 1 else 30
  dense_units = int(sys.argv[2]) if len(sys.argv) > 2 else 30

  train_path = os.environ.get("TRAIN_PATH")
  test_path = os.environ.get("TEST_PATH")

  with mlflow.start_run():
    mlflow.log_param("window_size", 60)
    mlflow.log_param("lstm_units", lstm_units)
    mlflow.log_param("num_lstm_layers", 2)
    mlflow.log_param("dense_units", dense_units)
    mlflow.log_param("learning_rate", 1.0000e-03)

    train_set = tf.data.Dataset.load(train_path)
    test_set = tf.data.Dataset.load(test_path)

    model = tf.keras.models.Sequential([
      tf.keras.layers.LSTM(lstm_units, input_shape=(60,1), return_sequences=True),
      tf.keras.layers.LSTM(lstm_units),
      tf.keras.layers.Dense(dense_units, activation="relu"),
      tf.keras.layers.Dense(1),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1.0000e-03)

    model.compile(loss=tf.keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])

    history = model.fit(train_set, 
                        validation_data=test_set,
                        epochs=32)

    train_mae = history.history['mae'][-1]
    test_mae = history.history['val_mae'][-1]

    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.keras.log_model(model, "lstm_model")
