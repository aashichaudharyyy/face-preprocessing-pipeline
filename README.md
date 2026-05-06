# RetinaFace Detection & Alignment

A Jupyter notebook demonstrating face detection and geometric alignment using the **RetinaFace** deep learning model.

---

## Overview

This project uses RetinaFace to:
1. **Detect multiple faces** in a group image and draw bounding boxes around them.
2. **Align a tilted face** by computing the rotation angle from eye landmark positions and correcting it.

---

## Requirements

Install the dependencies before running the notebook:

```bash
pip install retina-face opencv-python matplotlib numpy pillow
```

| Package | Purpose |
|---|---|
| `retina-face` | Face detection & landmark extraction |
| `opencv-python` | Image reading and rectangle drawing |
| `matplotlib` | Displaying images inline |
| `numpy` | Math operations |
| `Pillow` | Image rotation |

---

## Input Files

| File | Description |
|---|---|
| `group_img.png` | A photo containing multiple faces for group detection |
| `tilted_img.png` | A photo with a tilted/rotated face for alignment demo |

---

## How It Works

### 1. Multi-Face Detection (`group_img.png`)

- Loads the image using OpenCV.
- Calls `RetinaFace.detect_faces()` to identify all faces and their bounding boxes.
- Draws white rectangles around each detected face.
- Displays the annotated image via Matplotlib.

### 2. Face Alignment (`tilted_img.png`)

- Detects the face and extracts **eye landmark coordinates** (left eye and right eye).
- Computes the **tilt angle** using trigonometry:
  - Calculates the horizontal and vertical distance between eyes.
  - Uses the law of cosines to find the angle α.
  - Converts from radians to degrees.
- Rotates the image by `-α` degrees using Pillow to produce an upright face.
- Draws a bounding box on the aligned result and displays it.

---

## Usage

```bash
jupyter notebook RetinaFaceDetection_Recognition.ipynb
```

Make sure `group_img.png` and `tilted_img.png` are in the same directory as the notebook before running.

---

## Key Concepts

- **RetinaFace** is a robust single-stage face detector that also returns 5-point facial landmarks (eyes, nose, mouth corners).
- **Face alignment** is a common preprocessing step in face recognition pipelines — it normalizes pose variation to improve downstream model accuracy.
- The angle is derived from the eye coordinates using the formula:

```
cos(α) = (b² + c² - a²) / (2bc)
```

where `a` = vertical eye distance, `b` = horizontal eye distance, `c` = hypotenuse.

---

## References

- [RetinaFace Paper (arXiv)](https://arxiv.org/abs/1905.00641)
- [serengil/retinaface (GitHub)](https://github.com/serengil/retinaface)
