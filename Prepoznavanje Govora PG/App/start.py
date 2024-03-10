import tkinter as tk
import random
import tkinter as tk
from threading import Thread
from recorder import VoiceRecorder
import io
import soundfile as sf
import librosa
from drawingApp import DrawingApp
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import os

label_encoder = LabelEncoder()

def ucitaj_wav_fajlove(direktorijum):
    audio_podaci = {}
    for fajl in os.listdir(direktorijum):
        if fajl.endswith(".wav"):
            putanja = os.path.join(direktorijum, fajl)
            audio, sr = librosa.load(putanja, sr=None, mono=True)
            audio, _ = librosa.effects.trim(audio)

            # Normalizacija glasnoće
            audio = librosa.util.normalize(audio)

            # Filtriranje niskih frekvencija (opciono)
            audio = librosa.effects.preemphasis(audio)

            audio_podaci[fajl] = (audio, sr)
    return audio_podaci

def pad_matrix_to_fixed_size(matrix, target_shape=(20, 380)):
    if matrix.shape[1] > target_shape[1]:
        raise ValueError("Matrix width is greater than target width.")

    padding_width = target_shape[1] - matrix.shape[1]
    padded_matrix = np.pad(matrix, ((0, 0), (0, padding_width)), mode='constant')
    
    return padded_matrix

def normalize_mfcc(mfcc_coefficients):
    # Standardizacija: oduzimanje srednje vrednosti i deljenje sa standardnom devijacijom
    mfcc_mean = np.mean(mfcc_coefficients, axis=0)
    mfcc_std = np.std(mfcc_coefficients, axis=0)

    normalized_mfcc = (mfcc_coefficients - mfcc_mean) / mfcc_std
    return normalized_mfcc

def izracunaj_mfcc(audio, sr, n_mfcc=128):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return pad_matrix_to_fixed_size(normalize_mfcc(mfccs))


# GUI setup
def start_recording():
    recorder.start_recording()
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)

def evaluate_model(h5_file_path, test_data):
    # Učitavanje modela
    model = tf.keras.models.load_model(h5_file_path)
    test_data = test_data.reshape(1, 128, 380, 1)
    predictions = model.predict(test_data)

    # Pretvaranje predviđanja u konkretne oznake
    # Ovo će zavisiti od toga kako su vaše oznake kodirane
    predicted_labels = np.argmax(predictions, axis=1)

    # Iteracija kroz svaki uzorak u testnom skupu i prikazivanje predviđanja
    for i, prediction in enumerate(predicted_labels):
        print(prediction)
        if(prediction==0):
            app.process_command("izbrisi")
        elif(prediction==1):
            app.process_command("krug")
        elif(prediction==2):
            app.process_command("kvadrat")
        elif(prediction==3):
            app.process_command("oboji")
        elif(prediction==4):
            app.process_command("trougao")


def stop_recording():
    recorder.stop_recording()
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    # Handle the audio data here
    audio_data = recorder.convert_to_mfcc()

  

    mfcc_koeficijenti = pad_matrix_to_fixed_size(normalize_mfcc(audio_data))
    tagovi=['izbrisi','krug','kvadrat','oboji','trougao']
    label_encoder.fit(tagovi)
    integer_encoded_test = label_encoder.transform(tagovi)
    y_train_encoded = to_categorical(integer_encoded_test, num_classes=5)
    evaluate_model('y.h5',mfcc_koeficijenti)

# Kreiranje Tkinter root prozora
root = tk.Tk()
app = DrawingApp()
recorder = VoiceRecorder()

app.canvas.master.title("Crtanje Figura")
app.canvas.master.geometry("400x500")  # Dodavanje dimenzija prozora

button_frame = tk.Frame(root)
button_frame.pack(side=tk.TOP, pady=20)  # pady dodaje vertikalni razmak za centriranje

# Kreiranje dugmića unutar okvira
start_button = tk.Button(button_frame, text="Start Recording", command=start_recording)
stop_button = tk.Button(button_frame, text="Stop Recording", command=stop_recording, state=tk.DISABLED)

# Postavljanje dugmića unutar okvira koristeći pack sa side=tk.LEFT
start_button.pack(side=tk.LEFT, padx=10)  # padx dodaje horizontalni razmak između dugmića
stop_button.pack(side=tk.LEFT)

root.mainloop()
