import tensorflow
import numpy as np
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

digits_data = np.vstack([train_data, test_data])
digits_labels = np.hstack([train_labels, test_labels])

index = np.random.randint(0, digits_data.shape[0])
plt.imshow(digits_data[index], cmap='gray')
plt.title('Class: ' + str(digits_labels[index]));

# zip_object = zipfile.ZipFile(file=r'C:\Users\201311\Downloads\archive(1)\A_Z Handwritten Data\A_Z Handwritten Data.csv',
#                              mode='r')
# zip_object.extractall('./')
# zip_object.close()

dataset_az = pd.read_csv(r'C:\Users\201311\Downloads\archive(1)\A_Z_Handwritten_Data\A_Z_Handwritten_Data.csv').astype(
    'float32')

alphabet_data = dataset_az.drop('0', axis=1)
alphabet_labels = dataset_az['0']

alphabet_data = np.reshape(alphabet_data.values, (alphabet_data.shape[0], 28, 28))

index = np.random.randint(0, alphabet_data.shape[0])
plt.imshow(alphabet_data[index], cmap='gray')
plt.title('Class: ' + str(alphabet_labels[index]));

data = np.vstack([alphabet_data, digits_data])
labels = np.hstack([alphabet_labels, digits_labels])

data = np.array(data, dtype='float32')
data = np.expand_dims(data, axis=-1)

le = LabelBinarizer()
labels = le.fit_transform(labels)

plt.imshow(data[0].reshape(28, 28), cmap='gray')
plt.title(str(labels[0]));

classes_total = labels.sum(axis=0)

classes_weights = {}
for i in range(0, len(classes_total)):
    # print(i)
    classes_weights[i] = classes_total.max() / classes_total[i]

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1, stratify=labels)

augmentation = ImageDataGenerator(rotation_range=10, zoom_range=0.05, width_shift_range=0.1,
                                  height_shift_range=0.1, horizontal_flip=False)

network = Sequential()

network.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
network.add(MaxPool2D(pool_size=(2, 2)))

network.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
network.add(MaxPool2D(pool_size=(2, 2)))

network.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
network.add(MaxPool2D(pool_size=(2, 2)))

network.add(Flatten())

network.add(Dense(64, activation='relu'))
network.add(Dense(128, activation='relu'))

network.add(Dense(36, activation='softmax'))

network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

network.summary()

name_labels = '0123456789'
name_labels += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
name_labels = [l for l in name_labels]

print(name_labels)

file_model = 'custom_ocr.model'
epochs = 20
batch_size = 128

checkpointer = ModelCheckpoint(file_model, monitor='val_loss', verbose=1, save_best_only=True)

len(X_train) // batch_size

history = network.fit(augmentation.flow(X_train, y_train, batch_size=batch_size),
                      validation_data=(X_test, y_test),
                      steps_per_epoch=len(X_train) // batch_size, epochs=epochs,
                      class_weight=classes_weights, verbose=1, callbacks=[checkpointer])

predictions = network.predict(X_test, batch_size=batch_size)

network.evaluate(X_test, y_test)

print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=name_labels))

history.history.keys()

plt.plot(history.history['val_loss']);
plt.plot(history.history['val_accuracy']);

network.save('network', save_format='h5')

loaded_network = load_model(
    r'C:\Users\201311\Downloads\archive(1)\A_Z_Handwritten_Data\A_Z_Handwritten_Data.csv - recursos/OCR with Python/Models/network')
