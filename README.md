# RetinaFace — Advanced Implementation Guide

This repository provides an advanced implementation of **RetinaFace**, a state-of-the-art single-stage face localization method. Beyond basic detection, this project demonstrates high-level computer vision workflows including manual face alignment, occlusion testing, and multi-scale analysis.

---

## What is RetinaFace?

RetinaFace is a pixel-wise face localization method proposed by Deng et al. (2019) from Imperial College London. It is significantly more than a standard face detector; it simultaneously predicts:
1. **Face Bounding Boxes**
2. **5 Facial Landmarks** (eyes, nose, mouth corners)
3. **Dense 3D Face Vertices** (optional)

The name "Retina" refers to the model's ability to perform **multi-scale processing**, similar to how the human retina detects objects at different scales simultaneously before passing information to the brain.



### How It Works
1.  **Multi-scale Feature Extraction (FPN Backbone):** Uses a Feature Pyramid Network (typically ResNet-50) to produce feature maps at different resolutions (P2 to P5). High-resolution maps catch tiny faces; low-resolution maps catch large ones.
2.  **Context Module (SSH/DCN):** Uses Deformable Convolutional Networks to aggregate surrounding spatial info, helping the model "see" through occlusion or extreme poses.
3.  **Multi-task Loss Head:** A single shared head performs classification, box regression, and landmark regression in parallel.
4.  **Single Forward Pass:** All detection and landmark localization happen in one shot—no secondary refinement stages are required.

---

## Why use RetinaFace for Deepfake Detection?

In deepfake detection, precision is everything. RetinaFace has become the industry standard because:
* **Precision Alignment:** The 5 landmarks enable **Affine Alignment**, warping faces into a canonical frontal pose. This allows forgery classifiers to focus on pixel-level artifacts rather than learning to handle different head tilts.
* **Occlusion Robustness:** Deepfakes often involve hands passing in front of faces or partial obstructions. RetinaFace maintains high accuracy in these "in the wild" conditions.
* **Consistency:** It produces cleaner crops than older methods (like MTCNN), reducing background noise that could confuse a forgery classifier.

---

## What I Implemented in This Repo

This repository focuses on three core proofs of concept that demonstrate why RetinaFace is superior for advanced vision tasks:

### 1. 5-Point Landmark Detection & Alignment
The implementation detects all faces in an image and extracts the coordinates for the eyes, nose, and mouth corners. This is used to calculate a rotation angle ($\alpha$) and perform an affine transform to normalize the face pose.


### 2. Occlusion Robustness Testing
I implemented a specialized "Stress Test" script that synthetically occludes up to ~50% of the face (masking the lower half, upper half, or side). This proves that RetinaFace's context modules can successfully infer the location of hidden landmarks based on visible facial features.

### 3. Multi-Scale Detection in a Single Pass
The implementation includes a canvas test where the same face is resized to 5 different scales (from 24px to 400px wide). A single `detect_faces()` call identifies every face on the canvas simultaneously, demonstrating the efficiency of the Feature Pyramid Network.

---

## Project Structure
```text
.
├── RetinaFaceDetection&R...ipynb      # Main Notebook (Alignment & Stress Tests)
├── RetinaFaceDetection.py             # Modular detection script
├── group_img.png                      # Multi-face test image
├── tilted_img.png                     # Face alignment test image
└── requirements.txt                   # Dependency list
```
