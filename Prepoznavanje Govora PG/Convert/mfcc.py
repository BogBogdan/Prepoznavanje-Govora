import librosa
import numpy as np

def izracunaj_mfcc(audio, sr, n_mfcc=128):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs

def normalize_mfcc(mfcc_coefficients):
    # Standardizacija: oduzimanje srednje vrednosti i deljenje sa standardnom devijacijom
    mfcc_mean = np.mean(mfcc_coefficients, axis=0)
    mfcc_std = np.std(mfcc_coefficients, axis=0)

    normalized_mfcc = (mfcc_coefficients - mfcc_mean) / mfcc_std
    return normalized_mfcc