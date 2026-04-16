# Deep Learning-based Facial Emotion Recognition (FER-2013 + Flask)

## Objective
This course project builds a beginner-friendly facial emotion recognition system for AI in Digital Media.
The objective is to:
- train a small CNN on FER-2013 style data,
- evaluate model behavior with clear visual results,
- provide single-image prediction through CLI and Flask UI.

## Method
### 1) Dataset pipeline
- Data format: `ImageFolder` (`data/train`, `data/val`)
- Transform: grayscale, resize to `48x48`, normalize

### 2) Model
- A simple PyTorch CNN baseline (`EmotionCNN`) with small convolution blocks
- Designed for readability and fast course-level experiments

### 3) Training and evaluation
- Train/validation loop with epoch metrics (loss + accuracy)
- Best model saved to `models/emotion_cnn.pth`
- Evaluation visuals saved to `results/`:
  - `training_curves.png`
  - `confusion_matrix.png`

## Results (What to Present)
After running training, report:
- best validation accuracy
- training/validation loss and accuracy curves
- confusion matrix insights (which emotions are confused)

> Tip: Replace this section with your actual scores and screenshots before final submission.

## Conclusion
This project provides a clean baseline for FER-2013 emotion recognition and a simple Flask demo.
It is intentionally small, readable, and easy to extend in later iterations.

---

## Project Structure
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
├── PRESENTATION_OUTLINE.md
├── models/                  # saved model weights
├── results/                 # curves and confusion matrix images
├── uploads/                 # uploaded images from web UI
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Setup
```bash
pip install -r requirements.txt
```

## Dataset Layout
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

## Run
### Train
```bash
python train.py --data_dir data --epochs 10 --batch_size 64 --lr 0.001
```

### Predict single image
```bash
python predict.py --image path/to/your_image.jpg --model models/emotion_cnn.pth
```

### Start Flask app
```bash
python app.py
```
Then open: `http://127.0.0.1:5000`

## Presentation Outline
See: `PRESENTATION_OUTLINE.md`
