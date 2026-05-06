# 🧠 Face Preprocessing Pipeline (Detection + Alignment + Recognition)

This project implements a complete face preprocessing pipeline using **RetinaFace**, **OpenCV**, and **NumPy**. It detects faces, extracts facial landmarks, aligns tilted faces using eye coordinates, and prepares images for further tasks like face recognition or deep learning models.

---

## ✨ Features

* 🔍 Face Detection using RetinaFace
* 👁️ Facial Landmark Extraction (eyes, nose, mouth)
* 📐 Face Alignment using eye coordinates (atan2-based rotation)
* 🖼️ Image rotation with proper scaling (`expand=True`)
* 🎯 Bounding box visualization
* ⚡ Clean preprocessing pipeline for ML/DL tasks

---

## 🛠️ Tech Stack

* Python
* OpenCV
* RetinaFace
* NumPy
* Matplotlib

---

## 📂 Project Structure

```
.
├── tilted_img.png
├── RetinaFaceDetection&Reco.ipynb
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1. Detect faces using RetinaFace
2. Extract eye coordinates from facial landmarks
3. Compute tilt angle using:

```
alpha = np.degrees(np.arctan2(dy, dx))
```

4. Rotate image using:

```
aligned_img = Image.rotate(-alpha, expand=True)
```

5. (Optional) Draw bounding boxes

---

## 📸 Sample Output

* Detects tilted face
* Aligns it horizontally
* Prepares it for recognition models

(Add before/after images here)

---

## 🚀 Future Improvements

* [ ] Crop aligned face automatically
* [ ] Remove black borders after rotation
* [ ] Handle multiple faces
* [ ] Add face recognition (FaceNet / DeepFace)
* [ ] Build a simple web UI (Streamlit)
* [ ] Batch processing for datasets

---

## 💡 Use Cases

* Face recognition preprocessing
* Deepfake pipelines
* Computer vision projects
* Dataset cleaning

---

## 🤝 Contributing

Feel free to fork this repo and improve it!

---

## ⭐ If you like this project, give it a star!
