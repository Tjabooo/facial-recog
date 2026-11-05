# facial-recog

A school project (made together with @ciph0n) that trains a small convolutional neural network (Keras) to recognize which of five K-pop artists (members of ITZY) is in a photo. Faces are found with MTCNN / OpenCV cascades, cropped to 64x64 grayscale and classified.

Status: finished school project, not maintained. The dataset consists of faces cropped from images found online and is only included for the exercise.

## Contents
- `facial_recog.py`: the newer train-and-recognize script (MTCNN detection, 64x64 input, 70 epochs).
- `facial_recognition.py`: an earlier version of the same idea, kept for reference.
- `scraper.py`: downloads a few images per name with `google_images_download` (the author's comment: "barely works").
- `img_cropper.py`: crops faces from `known_faces/<name>/` into `cropped_faces/<name>/`.
- `rename.py`, `convert.py`: helpers for renaming files and converting formats to `.jpg` (the latter needs ImageMagick's `magick`).
- `cropped_faces/`: cropped training images, one folder per person.
- `face_labels.npy`: the label order used by the trained model. The trained model itself (`face_model.h5`) is not in the repo.
- `lib/`: OpenCV Haar/LBP cascade files.

## Requirements
Python 3 with: `numpy`, `opencv-python`, `tensorflow`/`keras`, `scikit-learn`, `imutils`, `mtcnn`, and optionally `google_images_download` for the scraper. No pinned versions are recorded, so you may need to adjust for current Keras/TensorFlow releases.

## Usage
Run from the repository root, since paths are relative.
```bash
python3 facial_recog.py
```
- If `face_model.h5` and `face_labels.npy` exist, the script loads them and runs recognition. By default it reads a single image whose path is hard-coded in the script (`old_code/known_faces/...`, which is not in the repo); edit that path, or set `input_type = 'c'` to use the webcam.
- Otherwise it trains on `cropped_faces/` and saves `face_model.h5` and `face_labels.npy`.
