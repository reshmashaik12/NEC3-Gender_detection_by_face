import os
import urllib.request

import cv2
import numpy as np

GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"
GENDER_PROTO_URL = (
    "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"
)
GENDER_MODEL_URL = (
    "https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=1"
)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

_gender_net = None

FEMALE_NAME_ENDINGS = (
    "a", "i", "ya", "vi", "ni", "tha", "ika", "ini", "wari", "valli", "devi", "bai",
)
MALE_NAME_ENDINGS = (
    "an", "ar", "esh", "raj", "dev", "han", "man", "esh", "kumar", "nath", "deep",
)


def _download_if_missing(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading gender model file: {path}")
    urllib.request.urlretrieve(url, path)


def _get_gender_net():
    global _gender_net
    if _gender_net is not None:
        return _gender_net
    _download_if_missing(GENDER_PROTO_URL, GENDER_PROTO)
    _download_if_missing(GENDER_MODEL_URL, GENDER_MODEL)
    _gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    return _gender_net


def predict_gender_from_image(face_bgr):
    """Predict gender from a BGR face crop using OpenCV's gender DNN."""
    if face_bgr is None or face_bgr.size == 0:
        return None, 0.0

    net = _get_gender_net()
    blob = cv2.dnn.blobFromImage(
        face_bgr, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    net.setInput(blob)
    preds = net.forward()[0]
    idx = int(np.argmax(preds))
    labels = ("Male", "Female")
    confidence = float(preds[idx])
    return labels[idx], confidence


def predict_gender_from_name(name):
    """Predict gender from a person's name using heuristics and gender-guesser."""
    if not name or name in ("Unknown", "Low Confidence"):
        return None, 0.0

    first_name = name.strip().split()[0]

    try:
        import gender_guesser.detector as gender

        detector = gender.Detector(case_sensitive=False)
        guess = detector.get_gender(first_name)
        mapping = {
            "male": ("Male", 0.85),
            "mostly_male": ("Male", 0.7),
            "female": ("Female", 0.85),
            "mostly_female": ("Female", 0.7),
            "andy": (None, 0.0),
            "unknown": (None, 0.0),
        }
        if guess in mapping:
            label, conf = mapping[guess]
            if label:
                return label, conf
    except ImportError:
        pass

    lower = first_name.lower()
    for ending in FEMALE_NAME_ENDINGS:
        if lower.endswith(ending) and len(lower) > len(ending) + 1:
            return "Female", 0.55
    for ending in MALE_NAME_ENDINGS:
        if lower.endswith(ending) and len(lower) > len(ending) + 1:
            return "Male", 0.55

    return None, 0.0


def predict_gender(face_bgr, name):
    """
    Combine image-based and name-based gender signals.
    Returns a display string for the UI.
    """
    img_gender, img_conf = predict_gender_from_image(face_bgr)
    name_gender, name_conf = predict_gender_from_name(name)

    if img_gender and name_gender:
        if img_gender == name_gender:
            conf = max(img_conf, name_conf)
            return f"Gender: {img_gender} ({conf * 100:.0f}%)"
        return (
            f"Gender: {img_gender} (img {img_conf * 100:.0f}%) / "
            f"{name_gender} (name {name_conf * 100:.0f}%)"
        )

    if img_gender:
        return f"Gender: {img_gender} ({img_conf * 100:.0f}%)"

    if name_gender:
        return f"Gender: {name_gender} ({name_conf * 100:.0f}%, name)"

    return "Gender: Unknown"