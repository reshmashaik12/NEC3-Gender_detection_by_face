# Gender Name Detection

A simple face capture, training, and recognition project using OpenCV and TensorFlow/Keras.

## Setup

1. Activate the virtual environment in PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Workflow

### 1. Capture face images

Use `capture_faces.py` to collect labeled face images.

```powershell
python capture_faces.py --name Alice --gender Female --num 50 --camera 0
```

- The script prompts for name and gender if not provided.
- Press `q` to quit early once enough images are captured.
- Images and metadata are stored in `data/registered_faces/<name>/`.

### 2. Train the model

Use `train_model.py` to train a recognition model from captured images.

```powershell
python train_model.py
```

This saves the model to `models/face_model.keras` and the label map to `label_map.json`.

> Note: Training requires at least two labeled person folders under `data/registered_faces/`.

### 3. Recognize faces

Use `recognize_faces.py` to detect and recognize faces from the webcam.

```powershell
python recognize_faces.py
```

- Press `q` to close the camera window.
- The script also displays a gender label derived from the face crop and the recognized name.

## Notes

- If the webcam does not open, try a different camera index with `--camera 1` or `--camera 2`.
- Make sure `data/registered_faces/` contains one folder per person for training.
