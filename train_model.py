import argparse
import json
import os
import sys

import numpy as np

try:
    import tensorflow as tf
    keras = tf.keras
except ImportError:
    try:
        import keras
    except ImportError as exc:
        raise ImportError("TensorFlow or Keras is required to run this script.") from exc
    keras = keras

DATA_DIR = "data/registered_faces"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_model.keras")
LABEL_MAP_PATH = "label_map.json"
IMG_SIZE = (224, 224)


def get_image_dataset_from_directory(*args, **kwargs):
    if hasattr(keras.utils, "image_dataset_from_directory"):
        return keras.utils.image_dataset_from_directory(*args, **kwargs)
    if hasattr(keras.preprocessing, "image_dataset_from_directory"):
        return keras.preprocessing.image_dataset_from_directory(*args, **kwargs)
    raise RuntimeError("image_dataset_from_directory is not available in this Keras installation.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a face recognition model from saved face images.")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory containing labeled face subfolders.")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Directory to save the trained model.")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to save the trained model.")
    parser.add_argument("--label-map", default=LABEL_MAP_PATH, help="Path to save the label map JSON file.")
    parser.add_argument("--img-size", nargs=2, type=int, default=IMG_SIZE, help="Training image width and height.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data to reserve for validation.")
    return parser.parse_args()


def build_model(num_classes, input_shape):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Rescaling(1.0 / 255.0),
        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train(args):
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")

    ensure_dir(args.model_dir)

    print("Loading dataset...")
    try:
        train_ds = get_image_dataset_from_directory(
            args.data_dir,
            image_size=tuple(args.img_size),
            batch_size=args.batch_size,
            label_mode="categorical",
            validation_split=args.val_split,
            subset="training",
            seed=123,
        )
        val_ds = get_image_dataset_from_directory(
            args.data_dir,
            image_size=tuple(args.img_size),
            batch_size=args.batch_size,
            label_mode="categorical",
            validation_split=args.val_split,
            subset="validation",
            seed=123,
        )
    except Exception as exc:
        raise RuntimeError("Failed to create image dataset from directory. Make sure the directory contains one subfolder per person.") from exc

    class_names = train_ds.class_names
    if len(class_names) < 1:
        raise ValueError("Need at least one labeled folder under the data directory to train a recognition model.")

    if len(class_names) == 1:
        print("Warning: only one person found. Recognition will only distinguish that person from the trained class.")

    label_map = {idx: name for idx, name in enumerate(class_names)}
    with open(args.label_map, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    model = build_model(len(class_names), input_shape=tuple(args.img_size) + (3,))
    model.summary()

    print("Starting training...")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    print(f"Saving model to {args.model_path}")
    model.save(args.model_path)
    print("Training complete.")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    args = parse_args()
    train(args)
   