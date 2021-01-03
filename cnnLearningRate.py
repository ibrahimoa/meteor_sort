import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing

def trainCNN( ):

    tf.keras.backend.clear_session()

    base_dir = 'G:\GIEyA\TFG\meteor_classification\labeledData\evenData'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'valid')

    train_meteors_dir = os.path.join(train_dir, 'meteors')
    train_non_meteors_dir = os.path.join(train_dir, 'non_meteors')
    validation_meteors_dir = os.path.join(validation_dir, 'meteors')
    validation_non_meteors_dir = os.path.join(validation_dir, 'non_meteors')

    print('total training meteors images: ', len(os.listdir(train_meteors_dir)))
    print('total training non-meteors images: ', len(os.listdir(train_non_meteors_dir)))
    print('total validation meteors images: ', len(os.listdir(validation_meteors_dir)))
    print('total validation non-meteors images: ', len(os.listdir(validation_non_meteors_dir)))


    #Rescale all images by 1./255

    train_datagen = ImageDataGenerator(rescale=1.0/255,
                                       rotation_range=10, #Range from 0 to 180 degrees to randomly rotate images
                                       width_shift_range=0.05, #Move image in this fram
                                       height_shift_range=0.05,
                                       #shear_range=0.2, #Girar la imagen
                                       zoom_range=0.1, #Zoom
                                       horizontal_flip=True, #Efecto cámara: girar la imagen con respecto al eje vertical
                                       fill_mode='nearest'
                                       ) #Ckeck other options

    test_datagen = ImageDataGenerator(rescale=1.0/255.)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=32,
                                                        class_mode='binary',
                                                        color_mode='grayscale',
                                                        target_size=(640, 360))
    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                            batch_size=32,
                                                            class_mode='binary',
                                                            color_mode='grayscale',
                                                            target_size=(640, 360))


    model = tf.keras.models.Sequential([#Try Dropout after each Conv2D + MaxPooling2D stage
        tf.keras.layers.Conv2D(16, (2,2), activation='relu', input_shape=(640, 360, 1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(16, (2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(16, (2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(12, (2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(12, (2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(12, (2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(432, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])

    print(model.summary())
    # Go from 1e-8 to 1e-3:
    lr_scheule = LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 10))
    # We are going to try different optimizers:
    optimizer = Adam(lr=1e-3)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=987, #4935
                        epochs=20, #Later train with more epochs if neccessary
                        validation_steps=247, #1234
                        callbacks=[lr_scheule])

    # loss per epoch vs lr per epoch:
    lrs = 1e-3 * (10 ** (np.arange(20) / 10))
    plt.figure(figsize=(12, 8))
    plt.semilogx(lrs, history.history["loss"])
    plt.axis([1e-3, 1e-1, 0, 1])  # Lowest value in the curve is approx 3e-3 -> Ideal lr
    plt.show()

    # Ideal lr values:
    # Adam : 1e-3 (For now ...)
    # SGD :
    # RMSProp :

if __name__ == '__main__':
    p = multiprocessing.Process(target=trainCNN)
    p.start()
    p.join()