import matplotlib.pyplot as plt
from Convert import mfcc
import numpy as np
import ucitaj
from cnnModel import model
from cnnModel import early_stopping
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

def pad_matrix_to_fixed_size(matrix, target_shape=(20, 380)):
    if matrix.shape[1] > target_shape[1]:
        raise ValueError("Matrix width is greater than target width.")

    padding_width = target_shape[1] - matrix.shape[1]
    padded_matrix = np.pad(matrix, ((0, 0), (0, padding_width)), mode='constant')
    
    return padded_matrix

def izdvoji_deo_do_underscore(naziv_datoteke):
    dijelovi = naziv_datoteke.replace('-','_').split('_')
    if len(dijelovi) > 1:
        return dijelovi[0]
    else:
        return naziv_datoteke

def ucitajsve(direktorijum):
    wav_fajlovi = ucitaj.ucitaj_wav_fajlove(direktorijum)

    mfcc_rečnik = {}

    for ime, (audio, sr) in wav_fajlovi.items():
        mfcc_koeficijenti = mfcc.normalize_mfcc(mfcc.izracunaj_mfcc(audio, sr))
        mfcc_rečnik[ime] = pad_matrix_to_fixed_size(mfcc_koeficijenti)
       

    X=[]
    y=[]
    for ime, mfcc_koeficijenti in mfcc_rečnik.items():
        X.append(mfcc_koeficijenti)
        y.append(izdvoji_deo_do_underscore(ime))
        #print(f"Ime: { izdvoji_deo_do_underscore(ime)}")
        #print("LPC Koeficijenti:", mfcc_koeficijenti.shape)
        #print()  # Dodaje praznu liniju za bolju preglednost

    return np.array(X),np.array(y)

def treniraj_model():
    X_train,test_tekst = ucitajsve('audio/training')
    label_encoder.fit(test_tekst)
    integer_encoded_test = label_encoder.transform(test_tekst)
    print(integer_encoded_test)
    y_train_encoded = to_categorical(integer_encoded_test, num_classes=5)
    history = model.fit(X_train, y_train_encoded, epochs=20, batch_size=64, validation_split=0.1, callbacks=[early_stopping])

def testiraj_model():
    X_train,test_tekst = ucitajsve('audio/test')
    label_encoder.fit(test_tekst)
    integer_encoded_test = label_encoder.transform(test_tekst)
    y_train_encoded = to_categorical(integer_encoded_test, num_classes=5)
    # Testiranje modela na test skupu
    test_loss, test_accuracy = model.evaluate(X_train, y_train_encoded)
    print("Test accuracy: ", test_accuracy)
    print("Test loss: ", test_loss)

    # Predviđanja modela za testni skup
    predictions = model.predict(X_train)

    # Pretvaranje predviđanja u konkretne oznake
    # Ovo će zavisiti od toga kako su vaše oznake kodirane
    predicted_labels = np.argmax(predictions, axis=1)

    # Iteracija kroz svaki uzorak u testnom skupu i prikazivanje predviđanja
    for i, prediction in enumerate(predicted_labels):
        print(f"Uzorak {i}: Predviđena oznaka = {prediction}, Stvarna oznaka = {np.argmax(y_train_encoded[i])}")

    model.save('my_model.h5')
    
def run():
    treniraj_model()
    testiraj_model()

run()

