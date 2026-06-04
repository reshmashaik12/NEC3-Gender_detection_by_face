import argparse
import json
import os
import time

import cv2

DATA_DIR = os.path.join("data", "registered_faces")
VALID_GENDERS = {"male", "female", "other", "unknown"}


def normalize_text(value):
    return value.strip() if value else ""


def ensure_person_folder(person_name):
    save_path = os.path.join(DATA_DIR, person_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path


def save_metadata(save_path, person_name, gender):
    metadata = {
        "name": person_name,
        "gender": gender,
    }
    with open(os.path.join(save_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def choose_gender(gender):
    gender_value = normalize_text(gender).lower()
    if gender_value in VALID_GENDERS:
        return gender_value.capitalize()
    return "Unknown"


def capture_faces(person_name, gender=None, num_images=50, camera_index=0):
    save_path = ensure_person_folder(person_name)
    if gender:
        save_metadata(save_path, person_name, gender)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: cannot open camera index {camera_index}")
        return 0

    print(f"Capturing faces for {person_name} ({gender}). Please look at the camera and move slowly...")
    print("Press 'q' to stop early.")
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Camera frame not available. Stopping capture.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            file_name = os.path.join(save_path, f"face_{int(time.time() * 1000000)}.jpg")
            cv2.imwrite(file_name, face)
            count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"Captured: {count}/{num_images}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        cv2.imshow("Face Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Successfully captured {count} images for {person_name}.")
    print("Please run train_model.py to train the recognition model with this data.")
    return count


def parse_args():
    parser = argparse.ArgumentParser(description="Capture face images for a person")
    parser.add_argument("--name", "-n", required=False, help="Name for images (will prompt if omitted)")
    parser.add_argument("--gender", "-g", required=False, help="Gender label for this person")
    parser.add_argument("--num", "-m", type=int, default=50, help="Number of images to capture")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    name = normalize_text(args.name)
    if not name:
        try:
            name = normalize_text(input("Enter the name for this person: "))
        except EOFError:
            name = ""

    if not name:
        print("No name provided — aborting.")
        exit(1)

    gender = normalize_text(args.gender)
    if not gender:
        try:
            gender = normalize_text(input("Enter gender (Male/Female/Other/Unknown): "))
        except EOFError:
            gender = ""
    gender = choose_gender(gender)

    capture_faces(name, gender, args.num, args.camera)