# Music ML Project - Source Code

This directory contains the main Python scripts for the IRMAS instrument recognition project.

## Scripts

### 1. `explore_dataset.py`
A comprehensive script to explore and analyze the IRMAS dataset.

**Features:**
- Loads and displays dataset statistics
- Extracts basic audio features (MFCC, spectral features, etc.)
- Creates visualizations of dataset distribution
- Analyzes sample audio files from each instrument
- Generates audio waveform and spectrogram plots

**Usage:**
```bash
python src/explore_dataset.py
```

### 2. `train_model.py`
Trains machine learning models for instrument classification.

**Features:**
- Extracts comprehensive audio features
- Trains Random Forest and SVM classifiers
- Evaluates model performance with confusion matrices
- Saves trained models for later use
- Creates performance visualizations

**Usage:**
```bash
python src/train_model.py
```

### 3. `predict.py`
Predicts the instrument in an audio file using trained models.

**Features:**
- Loads pre-trained models
- Extracts features from new audio files
- Makes predictions with confidence scores
- Supports both Random Forest and SVM models

**Usage:**
```bash
python src/predict.py path/to/audio.wav
python src/predict.py path/to/audio.wav --model svm
python src/predict.py path/to/audio.wav --no-probabilities
```

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Explore the dataset:**
   ```bash
   python src/explore_dataset.py
   ```

3. **Train models:**
   ```bash
   python src/train_model.py
   ```

4. **Make predictions:**
   ```bash
   python src/predict.py data/train/pia/sample.wav
   ```

## Dataset Structure

The IRMAS dataset should be organized as follows:
```
data/
└── train/
    ├── cel/     # cello
    ├── cla/     # clarinet
    ├── flu/     # flute
    ├── gac/     # acoustic guitar
    ├── gel/     # electric guitar
    ├── org/     # organ
    ├── pia/     # piano
    ├── sax/     # saxophone
    ├── tru/     # trumpet
    ├── vio/     # violin
    └── voi/     # voice
```

## Features Extracted

The scripts extract the following audio features:
- **Basic features:** duration, spectral centroid, spectral rolloff, zero crossing rate
- **MFCC features:** 13 MFCC coefficients (mean and std)
- **Chroma features:** 12 chroma features (mean and std)
- **Tonnetz features:** 6 tonnetz features (mean and std)

## Output Files

- `../notebooks/dataset_distribution.png` - Dataset distribution visualizations
- `../notebooks/sample_features.csv` - Sample feature extraction results
- `../models/` - Trained models and scaler
- `../models/confusion_matrix.png` - Model performance visualization

## Next Steps

1. Experiment with different feature extraction methods
2. Try deep learning approaches (CNN, LSTM)
3. Implement data augmentation techniques
4. Add cross-validation for better model evaluation
5. Create a web interface for real-time prediction
