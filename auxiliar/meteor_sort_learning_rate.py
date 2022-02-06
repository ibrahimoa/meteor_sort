from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt


def meteor_sort_learning_rate(model: Sequential, train_dir: str, image_resolution: (int, int), batch_size: int,
                              epochs: int, steps_per_epoch: int) -> None:
    """
    Train the model given with three different optimizers:
        - Adam
        - RMSProp
        - SGD

    The learning rate goes from 1e-6 to 1e0 (1). Then plot the model training error (loss) with all three optimizers.

    :param model: model for which to optimize the learning rate
    :param train_dir: training directory
    :param image_resolution: image resolution to be used
    :param batch_size: batch size to be used in the training
    :param epochs: number of epochs to use in the training (for now it has been built to use 60 epochs)
    :param steps_per_epoch: the steps per epoch (defined also by the number of epochs and the batch_size parameters)
    :return:
    """
    tf.keras.backend.clear_session()

    # Rescale all images by 1./255
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=batch_size,
                                                        class_mode='binary',
                                                        color_mode='grayscale',
                                                        target_size=image_resolution)

    print(model.summary())
    # Go from 1e-6 to 1e0:
    lr_schedule = LearningRateScheduler(lambda epoch: 1e-6 * 10 ** (epoch / 10))

    # We are going to try different optimizers:
    optimizer_1 = Adam(lr=1e-6)
    optimizer_2 = RMSprop(lr=1e-6)
    optimizer_3 = SGD(lr=1e-6)

    model.compile(optimizer=optimizer_1,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history1 = model.fit(train_generator,
                         steps_per_epoch=steps_per_epoch,
                         epochs=epochs,
                         shuffle=True,
                         verbose=1,
                         callbacks=[lr_schedule])

    model.compile(optimizer=optimizer_2,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history2 = model.fit(train_generator,
                         steps_per_epoch=steps_per_epoch,
                         epochs=epochs,
                         shuffle=True,
                         verbose=1,
                         callbacks=[lr_schedule])

    model.compile(optimizer=optimizer_3,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history3 = model.fit(train_generator,
                         steps_per_epoch=steps_per_epoch,
                         epochs=epochs,
                         shuffle=True,
                         verbose=1,
                         callbacks=[lr_schedule])

    # loss per epoch vs lr per epoch:
    lrs = 1e-6 * (10 ** (np.arange(epochs) / 10))
    plt.semilogx(lrs, history1.history["loss"])
    plt.semilogx(lrs, history2.history["loss"])
    plt.semilogx(lrs, history3.history["loss"])
    plt.title('Training set error versus learning rate with different optimizers')
    plt.xlabel('Learning rate')
    plt.ylabel('Training set error')
    plt.legend(['Adam', 'RMSprop', 'SGD'])
    plt.show()
