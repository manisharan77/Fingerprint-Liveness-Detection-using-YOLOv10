# 🔍 Fingerprint Authenticity Detection using YOLOv10

A deep learning-based web application that detects whether a fingerprint is **real or fake** using YOLOv10 object detection, wrapped in a Flask web interface.

---

## 🚀 Demo

Upload a fingerprint image and instantly get a prediction with confidence score — distinguishing between **real** and **fake** fingerprints.

---

## 🧠 Model

- **Architecture:** YOLOv10n (nano) — fine-tuned for binary classification
- **Classes:** `fake`, `real`
- **Dataset:** 812 annotated fingerprint images (via Roboflow)
- **Training:** 20 epochs, batch size 16, image size 640×640
- **Format:** YOLOv8-compatible annotation format

### Dataset Augmentations
- Horizontal flip (50%)
- Random rotation (±15°)
- Random shear (±15°)
- Brightness adjustment (±25%)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | YOLOv10 (Ultralytics) |
| Backend | Flask (Python) |
| Frontend | HTML/CSS (Jinja2 templates) |
| Image Processing | OpenCV, PIL |

---

## 📁 Project Structure

```
├── app.py                  # Flask web application
├── data.yaml               # Dataset configuration
├── Final_training.ipynb    # Model training notebook
├── static/
│   ├── uploads/            # User-uploaded images
│   └── results/            # Detection result images
├── templates/
│   ├── home.html
│   ├── login.html
│   ├── register.html
│   ├── index.html          # Detection page
│   ├── charts.html
│   └── performance.html
├── best.pt                 # Trained model weights (not tracked in git)
└── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your trained model
Place your `best.pt` weights file in the root directory.

### 4. Run the app
```bash
python app.py
```

Then open your browser at `http://127.0.0.1:5000`

---

## 🔐 Features

- User **Registration & Login** system
- Secure image upload and processing
- Real-time YOLO detection with **confidence scores**
- Visual comparison of original vs. annotated result image
- Performance charts and metrics dashboard

---

## 📊 Results

The model was trained on a balanced dataset of real and fake fingerprint images and evaluated on held-out test data.

| Metric | Value |
|--------|-------|
| Classes | 2 (fake, real) |
| Dataset Size | 812 images |
| Input Resolution | 640×640 |
| Model | YOLOv10n |

---

## 📜 Dataset License

Dataset provided via [Roboflow Universe](https://universe.roboflow.com/hi-rzede/fp-4r2fn) under **CC BY 4.0**.

---

## 👤 B MANI SHARAN

> Built as part of a Major Project 2025







ihguh3huwh2hi2bv3u3v2v3uhe eve


eiib3b3ieb3buebebrhudbdr



iej3ueehh3behej