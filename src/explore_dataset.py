#!/usr/bin/env python3
"""
Simple script to explore the IRMAS dataset for instrument recognition.
This script loads audio files, extracts basic features, and provides dataset statistics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_DIR = Path("data/train")
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

def load_dataset_info():
    """Load basic information about the dataset."""
    print("=== IRMAS Dataset Information ===")
    print(f"Dataset path: {DATA_DIR}")
    print(f"Number of instrument classes: {len(INSTRUMENTS)}")
    
    # Count files per instrument
    instrument_counts = {}
    total_files = 0
    
    for instrument_code, instrument_name in INSTRUMENTS.items():
        instrument_path = DATA_DIR / instrument_code
        if instrument_path.exists():
            file_count = len(list(instrument_path.glob("*.wav")))
            instrument_counts[instrument_name] = file_count
            total_files += file_count
            print(f"{instrument_name:15} ({instrument_code:3}): {file_count:4} files")
        else:
            print(f"{instrument_name:15} ({instrument_code:3}): Directory not found")
    
    print(f"\nTotal training files: {total_files}")
    return instrument_counts, total_files

def load_sample_audio(instrument_code, sample_idx=0):
    """Load a sample audio file for a given instrument."""
    instrument_path = DATA_DIR / instrument_code
    audio_files = list(instrument_path.glob("*.wav"))
    
    if not audio_files:
        print(f"No audio files found for {instrument_code}")
        return None, None
    
    if sample_idx >= len(audio_files):
        sample_idx = 0
    
    file_path = audio_files[sample_idx]
    print(f"Loading: {file_path.name}")
    
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def extract_basic_features(y, sr):
    """Extract basic audio features."""
    features = {}
    
    # Basic audio properties
    features['duration'] = len(y) / sr
    features['sample_rate'] = sr
    features['n_samples'] = len(y)
    
    # Spectral features
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # MFCC features (first 13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}'] = np.mean(mfccs[i])
    
    return features

def visualize_dataset_distribution(instrument_counts):
    """Create visualizations of the dataset distribution."""
    plt.figure(figsize=(15, 5))
    
    # Bar plot of instrument counts
    plt.subplot(1, 3, 1)
    instruments = list(instrument_counts.keys())
    counts = list(instrument_counts.values())
    
    plt.bar(range(len(instruments)), counts)
    plt.xticks(range(len(instruments)), [inst.replace('_', '\n') for inst in instruments], rotation=45)
    plt.title('Number of Files per Instrument')
    plt.ylabel('Count')
    
    # Pie chart
    plt.subplot(1, 3, 2)
    plt.pie(counts, labels=[inst.replace('_', '\n') for inst in instruments], autopct='%1.1f%%')
    plt.title('Distribution of Instruments')
    
    # Box plot (if we have enough data)
    plt.subplot(1, 3, 3)
    plt.boxplot(counts)
    plt.title('Distribution of File Counts')
    plt.ylabel('Number of Files')
    
    plt.tight_layout()
    plt.savefig('notebooks/dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_sample_audio():
    """Analyze a sample audio file from each instrument."""
    print("\n=== Sample Audio Analysis ===")
    
    sample_features = []
    
    for instrument_code, instrument_name in INSTRUMENTS.items():
        print(f"\nAnalyzing {instrument_name} ({instrument_code})...")
        
        y, sr = load_sample_audio(instrument_code)
        if y is not None:
            features = extract_basic_features(y, sr)
            features['instrument'] = instrument_name
            features['instrument_code'] = instrument_code
            sample_features.append(features)
            
            print(f"  Duration: {features['duration']:.2f} seconds")
            print(f"  Sample rate: {features['sample_rate']} Hz")
            print(f"  Spectral centroid: {features['spectral_centroid']:.2f}")
            print(f"  Zero crossing rate: {features['zero_crossing_rate']:.4f}")
    
    return pd.DataFrame(sample_features)

def plot_audio_waveform(y, sr, title="Audio Waveform"):
    """Plot audio waveform."""
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

def plot_spectrogram(y, sr, title="Spectrogram"):
    """Plot audio spectrogram."""
    plt.figure(figsize=(12, 6))
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    
    # Plot spectrogram
    librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the dataset exploration."""
    print("Starting IRMAS Dataset Exploration...")
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"Error: Data directory {DATA_DIR} not found!")
        print("Please make sure the dataset is properly extracted.")
        return
    
    # Load dataset information
    instrument_counts, total_files = load_dataset_info()
    
    # Visualize dataset distribution
    print("\nCreating visualizations...")
    visualize_dataset_distribution(instrument_counts)
    
    # Analyze sample audio files
    sample_df = analyze_sample_audio()
    
    if not sample_df.empty:
        print("\n=== Sample Features Summary ===")
        print(sample_df[['instrument', 'duration', 'spectral_centroid', 'zero_crossing_rate']].to_string(index=False))
        
        # Save sample features to CSV
        sample_df.to_csv('notebooks/sample_features.csv', index=False)
        print(f"\nSample features saved to: notebooks/sample_features.csv")
    
    # Demonstrate audio visualization for one sample
    print("\n=== Audio Visualization Demo ===")
    sample_instrument = 'pia'  # piano
    y, sr = load_sample_audio(sample_instrument)
    
    if y is not None:
        plot_audio_waveform(y, sr, f"Sample {INSTRUMENTS[sample_instrument]} Audio")
        plot_spectrogram(y, sr, f"Sample {INSTRUMENTS[sample_instrument]} Spectrogram")
    
    print("\nDataset exploration complete!")
    print("Next steps:")
    print("1. Install required packages: pip install -r requirements.txt")
    print("2. Run this script: python src/explore_dataset.py")
    print("3. Check the generated visualizations in the notebooks/ folder")
    print("4. Start building your ML model!")

if __name__ == "__main__":
    main()
