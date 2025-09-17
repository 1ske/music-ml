#!/usr/bin/env python3
"""
Simple prediction script for instrument classification.
This script loads a trained model and predicts the instrument in an audio file.
"""

import numpy as np
import pandas as pd
import librosa
import joblib
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set up paths
MODELS_DIR = Path("models")

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
    (Same function as in train_model.py)
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

def load_model(model_name="random_forest"):
    """
    Load a trained model and scaler.
    
    Args:
        model_name: Name of the model to load ('random_forest' or 'svm')
    
    Returns:
        Tuple of (model, scaler)
    """
    model_path = MODELS_DIR / f"{model_name}.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} not found. Please train a model first.")
    
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file {scaler_path} not found. Please train a model first.")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def predict_instrument(audio_file, model_name="random_forest", show_probabilities=True):
    """
    Predict the instrument in an audio file.
    
    Args:
        audio_file: Path to audio file
        model_name: Name of the model to use
        show_probabilities: Whether to show prediction probabilities
    
    Returns:
        Predicted instrument and probabilities (if available)
    """
    print(f"Analyzing: {audio_file}")
    
    # Load model
    try:
        model, scaler = load_model(model_name)
        print(f"Loaded model: {model_name}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None
    
    # Extract features
    features = extract_features(audio_file)
    if features is None:
        return None, None
    
    # Convert to DataFrame
    feature_df = pd.DataFrame([features])
    
    # Remove any non-feature columns
    feature_columns = [col for col in feature_df.columns if col != 'instrument']
    X = feature_df[feature_columns]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    
    # Get probabilities if available
    probabilities = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_scaled)[0]
        classes = model.classes_
        probabilities = dict(zip(classes, proba))
    
    print(f"Predicted instrument: {prediction}")
    
    if show_probabilities and probabilities:
        print("\nPrediction probabilities:")
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for instrument, prob in sorted_probs:
            print(f"  {instrument}: {prob:.4f}")
    
    return prediction, probabilities

def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict instrument in audio file')
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--model', default='random_forest', 
                       choices=['random_forest', 'svm'],
                       help='Model to use for prediction')
    parser.add_argument('--no-probabilities', action='store_true',
                       help='Hide prediction probabilities')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file {audio_path} not found!")
        return
    
    # Make prediction
    prediction, probabilities = predict_instrument(
        args.audio_file, 
        args.model, 
        not args.no_probabilities
    )
    
    if prediction is not None:
        print(f"\nResult: The audio file contains a {prediction}")
    else:
        print("Prediction failed!")

if __name__ == "__main__":
    main()
