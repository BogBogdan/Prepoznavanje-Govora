import tkinter as tk
from threading import Thread
import pyaudio
import wave
from recorder import VoiceRecorder
import io
import soundfile as sf
import librosa

def izracunaj_mfcc(audio, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs
# GUI setup
def start_recording():
    recorder.start_recording()
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)

def stop_recording():
    recorder.stop_recording()
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    # Handle the audio data here
    audio_data = recorder.get_audio_data()
    print(type(audio_data))
    wavio = io.BytesIO()
    with sf.SoundFile(wavio, mode='w', samplerate=44100, channels=1, format='WAV') as file:
        file.write(audio_data)

    # Now we can use librosa to load from this in-memory file
    wavio.seek(0)  # Go back to the start of the in-memory file
    audio, sr = librosa.load(wavio, sr=None, mono=True)
    audio, _ = librosa.effects.trim(audio)
    audio = librosa.util.normalize(audio)
    audio = librosa.effects.preemphasis(audio)

    mfcc_koeficijenti = izracunaj_mfcc(audio,sr)
    
    # PRIMENITI MODEL ZA RACUNANJE 
    

root = tk.Tk()
root.title("Voice Recorder")

recorder = VoiceRecorder()

start_button = tk.Button(root, text="Start Recording", command=start_recording)
stop_button = tk.Button(root, text="Stop Recording", command=stop_recording, state=tk.DISABLED)

start_button.pack(side=tk.LEFT)
stop_button.pack(side=tk.LEFT)


root.mainloop()
