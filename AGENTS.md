
# AGENTS.md

## Project Goal
Build a deep learning-based facial emotion recognition system using FER-2013 dataset with a Flask web interface.

---

## Development Rules

### General
- Keep the code simple and beginner-friendly
- Use clear variable names and comments
- Avoid over-engineering
- Prefer readability over performance

### Model
- Start with a simple CNN baseline
- Do NOT use complex architectures unless explicitly requested
- Ensure model can train within reasonable time

### Dataset
- Assume dataset is organized using ImageFolder format
- Use grayscale transforms
- Normalize input data

### Training
- Print loss and accuracy for each epoch
- Save best model to: models/emotion_cnn.pth
- Keep training script easy to run

### Inference
- Must support single image prediction
- Return:
  - predicted label
  - class probabilities

### Flask App
- Simple UI only
- Allow image upload
- Display:
  - uploaded image
  - predicted emotion
  - probabilities

### File Structure
- models/ → trained weights
- templates/ → HTML
- static/ → CSS / images
- uploads/ → user uploads

### Documentation
- Always update README when adding new features
- Include run commands

---

## Coding Constraints
- Use Python only
- Use PyTorch
- Avoid unnecessary dependencies
- No GPU requirement

---

## Task Strategy
- First build minimal working version
- Then improve step by step
- Always ensure code runs before adding complexity

---

## Done Criteria
- Training script runs successfully
- Model saves and loads correctly
- Flask app works
- User can upload image and get prediction
