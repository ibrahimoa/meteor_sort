from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import multiprocessing
from tensorflow.keras.constraints import unit_norm
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from os import listdir


def trainCNN():
    tf.keras.backend.clear_session()

    tf.keras.backend.clear_session()

    base_dir = 'C:\work_dir\meteorData\extraData_85_15'  # We don't use filtered data ... Not so useful

    train_dir = join(base_dir, 'train')

    ImageResolution: tuple = (256, 256)  # (432, 432) | (300, 300) |
    ImageResolutionGrayScale: tuple = (256, 256, 1)  # (432, 432, 1) | (300, 300, 1)
    EPOCHS: int = 60

    training_images = len(listdir(join(train_dir, 'meteors'))) + len(listdir(join(train_dir, 'non_meteors')))
    batch_size: int = 64
    steps_per_epoch: int = int(training_images / batch_size / 4)

    # Rescale all images by 1./255

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=batch_size,
                                                        class_mode='binary',
                                                        color_mode='grayscale',
                                                        target_size=ImageResolution)

    model = tf.keras.models.Sequential([

        Conv2D(8, (7, 7), activation='elu', input_shape=ImageResolutionGrayScale,
               strides=1, kernel_initializer='he_uniform', kernel_constraint=unit_norm()),
        MaxPooling2D(pool_size=(3, 3)),
        BatchNormalization(),

        Conv2D(12, (5, 5), activation='elu', kernel_initializer='he_uniform', kernel_constraint=unit_norm()),
        MaxPooling2D(pool_size=(3, 3)),
        BatchNormalization(),

        Conv2D(12, (3, 3), activation='elu', kernel_initializer='he_uniform', kernel_constraint=unit_norm()),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_uniform', kernel_constraint=unit_norm()),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Flatten(),
        Dense(200, activation='elu', kernel_initializer='he_uniform', kernel_constraint=unit_norm()),
        BatchNormalization(),
        Dense(16, activation='elu', kernel_initializer='he_uniform', kernel_constraint=unit_norm()),
        BatchNormalization(),
        Dense(1, activation='sigmoid', kernel_initializer='he_uniform')
    ])

    print(model.summary())
    # Go from 1e-6 to 1e0:
    lr_scheule = LearningRateScheduler(lambda epoch: 1e-6 * 10 ** (epoch / 10))

    # We are going to try different optimizers:
    optimizer1 = Adam(lr=1e-6)
    optimizer2 = RMSprop(lr=1e-6)
    optimizer3 = SGD(lr=1e-6)

    model.compile(optimizer=optimizer1,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history1 = model.fit(train_generator,
                         steps_per_epoch=steps_per_epoch,
                         epochs=EPOCHS,
                         shuffle=True,
                         verbose=2,
                         callbacks=[lr_scheule])

    model.compile(optimizer=optimizer2,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history2 = model.fit(train_generator,
                         steps_per_epoch=steps_per_epoch,
                         epochs=EPOCHS,
                         shuffle=True,
                         verbose=2,
                         callbacks=[lr_scheule])

    model.compile(optimizer=optimizer3,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history3 = model.fit(train_generator,
                         steps_per_epoch=steps_per_epoch,
                         epochs=EPOCHS,
                         shuffle=True,
                         verbose=2,
                         callbacks=[lr_scheule])

    # loss per epoch vs lr per epoch:
    lrs = 1e-6 * (10 ** (np.arange(EPOCHS) / 10))
    plt.semilogx(lrs, history1.history["loss"])
    plt.semilogx(lrs, history2.history["loss"])
    plt.semilogx(lrs, history3.history["loss"])
    plt.title('Error en conjunto de entrenamiento frente a ratio de aprendizaje con distintos optimizadores')
    plt.xlabel('Ratio de aprendizaje')
    plt.ylabel('Error en el conjunto de entrenamiento')
    plt.legend(['Adam', 'RMSprop', 'SGD'])
    plt.show()


if __name__ == '__main__':
    p = multiprocessing.Process(target=trainCNN)
    p.start()
    p.join()
