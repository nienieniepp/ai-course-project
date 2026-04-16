# Deep Learning-based Facial Emotion Recognition (FER-2013 + Flask)

## Project Goal
This project is a beginner-friendly university AI for Digital Media course project.
It aims to:
- train a simple CNN model for facial emotion recognition using FER-2013,
- support single-image prediction,
- provide a basic Flask web interface for uploading an image and showing results.

The current version is an **initial skeleton** designed for the next step: dataset loading, training, and iterative improvements.

## Folder Structure
```text
ai-course-project/
├── app.py
├── train.py
├── predict.py
├── model.py
├── dataset.py
├── utils.py
├── requirements.txt
├── README.md
├── models/                 # saved model weights (emotion_cnn.pth)
├── uploads/                # uploaded images from web UI
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Setup
1. Create and activate a Python virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Expected Dataset Layout (next step)
Use ImageFolder format under `data/`:

```text
data/
├── train/
│   ├── Angry/
│   ├── Disgust/
│   └── ...
└── val/
    ├── Angry/
    ├── Disgust/
    └── ...
```

Images are transformed to grayscale, resized to 48x48, and normalized.

## Run Commands
### 1) Train baseline model
```bash
python train.py --data_dir data --epochs 10 --batch_size 64
```
Best model weights are saved to:

```text
models/emotion_cnn.pth
```

### 2) Predict one image from command line
```bash
python predict.py --image path/to/your_image.jpg
```
Outputs:
- predicted emotion label,
- class probabilities.

### 3) Start Flask web app
```bash
python app.py
```
Open in browser:

```text
http://127.0.0.1:5000
```

## Notes for Next Step
- Add real FER-2013 dataset files under `data/train` and `data/val`.
- Train and evaluate baseline CNN.
- Improve UI and evaluation reports gradually.
- Keep code simple and readable for course use.
