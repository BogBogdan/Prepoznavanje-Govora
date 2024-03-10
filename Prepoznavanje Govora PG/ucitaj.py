import os
import librosa


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

