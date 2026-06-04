import argparse
import json
import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from gender_predictor import predict_gender

# Configuration
DEFAULT_MODEL_PATH = "models/face_model.keras"
DEFAULT_LABEL_MAP_PATH = "label_map.json"
IMG_SIZE = (224, 224)
THRESHOLD = 0.5  # Confidence threshold
DATA_DIR = os.path.join("data", "registered_faces")

def load_label_map(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Label map not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata_for_name(name):
    metadata_path = os.path.join(DATA_DIR, name, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def recognize(model_path, label_map_path, camera_index):
    if not os.path.exists(model_path) or not os.path.exists(label_map_path):
        print("Model or Label Map not found. Please run train_model.py first.")
        return

    print("Loading model...")
    model = load_model(model_path)
    label_map = load_label_map(label_map_path)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera index {camera_index}.")
        return

    print("Starting webcam... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            y_start = y + int(h * 0.1)
            y_end = y + h - int(h * 0.1)
            x_start = x + int(w * 0.1)
            x_end = x + w - int(w * 0.1)
            face_img = frame[y_start:y_end, x_start:x_end]

            if face_img.size == 0:
                continue

            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img_resized = cv2.resize(face_img_rgb, IMG_SIZE)
            face_img_array = tf.keras.preprocessing.image.img_to_array(face_img_resized)
            face_img_array = np.expand_dims(face_img_array, axis=0)

            predictions = model.predict(face_img_array, verbose=0)
            score = float(np.max(predictions))
            class_idx = int(np.argmax(predictions))

            display_name = "Unknown"
            if score > THRESHOLD and str(class_idx) in label_map:
                display_name = label_map[str(class_idx)]
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            if display_name != "Unknown":
                metadata = load_metadata_for_name(display_name)
                gender_label = f"Gender: {metadata.get('gender', 'Unknown')}"
            else:
                gender_label = predict_gender(face_img, display_name)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{display_name} ({score*100:.1f}%)", (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, gender_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Recognize faces from webcam")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to the trained model")
    parser.add_argument("--label-map", default=DEFAULT_LABEL_MAP_PATH, help="Path to the label map JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    recognize(args.model_path, args.label_map, args.camera)
