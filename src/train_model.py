#!/usr/bin/env python3
"""
Simple training script for instrument classification using the IRMAS dataset.
This script demonstrates a basic machine learning pipeline for audio classification.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set up paths
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
# Set paths relative to the script location
DATA_DIR = SCRIPT_DIR / ".." / "data" / "train"
MODELS_DIR = SCRIPT_DIR / ".." / "models"
MODELS_DIR.mkdir(exist_ok=True)

INSTRUMENTS = {
    'cel': 'cello',
    'cla': 'clarinet', 
    'flu': 'flute',
    'gac': 'acoustic_guitar',
    'gel': 'electric_guitar',
    'org': 'organ',
    'pia': 'piano',
    'sax': 'saxophone',
    'tru': 'trumpet',
    'vio': 'violin',
    'voi': 'voice'
}

def extract_features(audio_file, sr=22050, n_mfcc=13):
    """
    Extract audio features from a single audio file.
    
    Args:
        audio_file: Path to audio file
        sr: Sample rate for loading audio
        n_mfcc: Number of MFCC coefficients to extract
    
    Returns:
        Dictionary of extracted features
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=sr)
        
        # Basic features
        features = {
            'duration': len(y) / sr,
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
            'rms_energy': np.mean(librosa.feature.rms(y=y)),
        }
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        for i in range(n_mfcc):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            features[f'chroma_{i+1}_mean'] = np.mean(chroma[i])
            features[f'chroma_{i+1}_std'] = np.std(chroma[i])
        
        # Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        for i in range(6):
            features[f'tonnetz_{i+1}_mean'] = np.mean(tonnetz[i])
            features[f'tonnetz_{i+1}_std'] = np.std(tonnetz[i])
        
        return features
        
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

def load_dataset(max_files_per_instrument=50):
    """
    Load the IRMAS dataset with feature extraction.
    
    Args:
        max_files_per_instrument: Maximum number of files to load per instrument (for faster training)
    
    Returns:
        Tuple of (features_df, labels)
    """
    print("Loading IRMAS dataset...")
    
    all_features = []
    all_labels = []
    
    for instrument_code, instrument_name in INSTRUMENTS.items():
        print(f"Processing {instrument_name} ({instrument_code})...")
        
        instrument_path = DATA_DIR / instrument_code
        if not instrument_path.exists():
            print(f"  Warning: Directory {instrument_path} not found")
            continue
        
        # Get list of audio files
        audio_files = list(instrument_path.glob("*.wav"))
        
        # Limit number of files for faster training
        if len(audio_files) > max_files_per_instrument:
            audio_files = audio_files[:max_files_per_instrument]
        
        print(f"  Found {len(audio_files)} files")
        
        # Process each audio file
        for i, audio_file in enumerate(audio_files):
            if i % 10 == 0:
                print(f"    Processing file {i+1}/{len(audio_files)}")
            
            features = extract_features(audio_file)
            if features is not None:
                all_features.append(features)
                all_labels.append(instrument_name)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    features_df['instrument'] = all_labels
    
    print(f"\nDataset loaded: {len(features_df)} samples, {len(features_df.columns)-1} features")
    return features_df

def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and compare their performance.
    
    Args:
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing labels
    
    Returns:
        Dictionary of trained models
    """
    print("\nTraining models...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    
    models = {}
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf
    
    # SVM
    print("Training SVM...")
    svm = SVC(kernel='rbf', random_state=42, probability=True)
    svm.fit(X_train_scaled, y_train)
    models['SVM'] = svm
    
    return models, scaler

def evaluate_models(models, X_test, y_test, scaler):
    """
    Evaluate trained models and create visualizations.
    
    Args:
        models: Dictionary of trained models
        X_test, y_test: Testing data
        scaler: Fitted scaler
    """
    print("\nEvaluating models...")
    
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name} Results:")
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    # Create confusion matrix for best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    y_pred_best = results[best_model_name]['predictions']
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(set(y_test)), 
                yticklabels=sorted(set(y_test)))
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def save_models(models, scaler):
    """Save trained models to disk."""
    print("\nSaving models...")
    
    for name, model in models.items():
        model_path = MODELS_DIR / f'{name.lower().replace(" ", "_")}.pkl'
        joblib.dump(model, model_path)
        print(f"Saved {name} to {model_path}")
    
    print(f"Scaler saved to {MODELS_DIR / 'scaler.pkl'}")

def main():
    """Main training function."""
    print("IRMAS Instrument Classification Training")
    print("=" * 50)
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"Error: Data directory {DATA_DIR} not found!")
        print("Please make sure the dataset is properly extracted.")
        return
    
    # Load dataset
    df = load_dataset(max_files_per_instrument=100)  # Limit for faster training
    
    if df.empty:
        print("Error: No data loaded!")
        return
    
    # Prepare features and labels
    feature_columns = [col for col in df.columns if col != 'instrument']
    X = df[feature_columns]
    y = df['instrument']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Number of features: {len(feature_columns)}")
    
    # Train models
    models, scaler = train_models(X_train, X_test, y_train, y_test)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test, scaler)
    
    # Save models
    save_models(models, scaler)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Training Summary:")
    for name, result in results.items():
        print(f"{name}: {result['accuracy']:.4f} accuracy")
    
    print(f"\nModels saved to: {MODELS_DIR}")
    print("Next steps:")
    print("1. Try different feature extraction methods")
    print("2. Experiment with deep learning models (CNN, LSTM)")
    print("3. Use the saved models for prediction on new audio files")

if __name__ == "__main__":
    main()
