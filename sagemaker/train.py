import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    # Sagemaker sets these env vars to the local paths where it downloaded
    # each S3 "channel" (train/test) inside the training container.
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    # Sagemaker's TensorFlow container always injects this flag as
    # "--model_dir" (underscore), regardless of what's in hyperparameters,
    # so the flag name has to match exactly, not "--model-dir".
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    return parser.parse_args()


def build_model(num_classes):
    # Same architecture as cnn_model (the winning variant, with pooling)
    # in the main notebook's section 3/4.
    model = Sequential([
        Conv2D(32, kernel_size=3, padding="same", activation="relu", input_shape=(48, 48, 1)),
        MaxPooling2D(2),
        Conv2D(64, kernel_size=3, padding="same", activation="relu"),
        MaxPooling2D(2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    args = parse_args()

    X_train = np.load(os.path.join(args.train, "X_train.npy"))
    y_train = np.load(os.path.join(args.train, "y_train.npy"))
    X_test = np.load(os.path.join(args.test, "X_test.npy"))
    y_test = np.load(os.path.join(args.test, "y_test.npy"))

    num_classes = int(y_train.max()) + 1
    model = build_model(num_classes)

    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")

    # I save to the LOCAL SM_MODEL_DIR path, not args.model_dir (which Sagemaker
    # sets to an S3 URI meant for TensorFlow's own distributed checkpointing,
    # not for plain model.save()). Sagemaker uploads this local folder to S3
    # for me once the script exits. "1" is the SavedModel version subfolder
    # the prebuilt TensorFlow Serving container expects.
    model.save(os.path.join(os.environ["SM_MODEL_DIR"], "1"))
