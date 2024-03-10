from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping

# Definisanje EarlyStopping callback-a
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
# Number of LPC coefficients and number of time steps
n_coefficients = 380  # You provided 20 coefficients in the example
n_timesteps = 128   # This is an assumption; you'll need to adjust based on your data

# Number of classes you have in your classification problem
n_classes = 5  # Adjust this based on your actual number of classes

model = Sequential()

# Prvi sloj konvolucije i max pooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(n_timesteps, n_coefficients, 1)))  # MFCC input će biti 2D tako da poslednja dimenzija mora biti 1
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

# Spljoštenje i gusti slojevi
model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='softmax'))  # Broj klasa


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])