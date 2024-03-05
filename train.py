from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import cv2
from keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping

store_image = []
train_path = r'D:\bureau\BD&AI 1\s4\algo\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Train'
fps = 5
train_images_path = os.path.join(train_path, 'frames')
os.makedirs(train_images_path, exist_ok=True)

def store_inarray(image_path):
    image = load_img(image_path)
    image = img_to_array(image)
    image = cv2.resize(image, (227, 227), interpolation=cv2.INTER_AREA)
    gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    store_image.append(gray)

# Check if images are loaded
for video in os.listdir(train_path):
    video_path = os.path.join(train_path, video)
    os.system(f'ffmpeg -i "{video_path}" -r 1/{fps} "{train_images_path}/%03d.jpg"')

images = os.listdir(train_images_path)
for image in images:
    image_path = os.path.join(train_images_path, image)
    store_inarray(image_path)

print("Number of images loaded:", len(store_image))

if len(store_image) > 0:
    # Process the loaded images
    store_image = np.array(store_image)
    store_image = (store_image - store_image.mean()) / (store_image.std())
    store_image = np.clip(store_image, 0, 1)

    # Save processed images
    np.save('training.npy', store_image)

    # Load processed images
    training_data = np.load('training.npy')
    frames = training_data.shape[2] if len(training_data.shape) >= 3 else 0

    if frames > 0:
        frames = frames - frames % 10
        training_data = training_data[:, :, :frames]
        training_data = training_data.reshape(-1, 227, 227, 10)
        training_data = np.expand_dims(training_data, axis=4)
        target_data = training_data.copy()

        epochs = 5
        batch_size = 1

        callback_save = ModelCheckpoint("saved_model.h5", monitor="val_loss", save_best_only=True)
        callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        stae_model = Sequential()

stae_model.add(Conv3D(filters=128, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding='valid',
                      input_shape=(227, 227, 10, 1), activation='tanh'))
stae_model.add(Conv3D(filters=64, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='valid', activation='tanh'))
stae_model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', dropout=0.4, recurrent_dropout=0.3,
                          return_sequences=True))
stae_model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', dropout=0.3, return_sequences=True))
stae_model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, return_sequences=True, padding='same',
                          dropout=0.5))
stae_model.add(Conv3DTranspose(filters=128, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='valid', activation='tanh'))
stae_model.add(Conv3DTranspose(filters=1, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding='valid', activation='tanh'))

stae_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        stae_model.fit(training_data, target_data, batch_size=batch_size, epochs=epochs,
                       callbacks=[callback_save, callback_early_stopping], validation_split=0.2)

        stae_model.save("saved_model.h5")
    else:
        print("Error: Not enough frames after processing.")
else:
    print("Error: No images loaded. Check the image loading process.")
