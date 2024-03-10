import tkinter as tk
from threading import Thread
import pyaudio
import librosa
import numpy as np

class VoiceRecorder:
    def __init__(self):
        self.is_recording = False
        self.frames = []
        self.sample_rate = 44100  # Dodato za librosa

    def start_recording(self):
        self.is_recording = True
        self.frames = []  # Reset the frames list
        self.thread = Thread(target=self.record)
        self.thread.start()

    def stop_recording(self):
        self.is_recording = False
        self.thread.join()

    def record(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = self.sample_rate
        CHUNK = 1024
        
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
        while self.is_recording:
            data = stream.read(CHUNK)
            self.frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

    def get_audio_data(self):
        return b''.join(self.frames)

    def convert_to_mfcc(self, n_mfcc=128):
        audio_data = np.frombuffer(self.get_audio_data(), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max  # Normalizacija
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=n_mfcc)
        return mfccs
