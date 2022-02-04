from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt


def meteorSortLearningRate(model, train_dir, image_resolution, batch_size, epochs, steps_per_epoch):
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
    optimizer1 = Adam(lr=1e-6)
    optimizer2 = RMSprop(lr=1e-6)
    optimizer3 = SGD(lr=1e-6)

    model.compile(optimizer=optimizer1,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history1 = model.fit(train_generator,
                         steps_per_epoch=steps_per_epoch,
                         epochs=epochs,
                         shuffle=True,
                         verbose=1,
                         callbacks=[lr_schedule])

    model.compile(optimizer=optimizer2,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history2 = model.fit(train_generator,
                         steps_per_epoch=steps_per_epoch,
                         epochs=epochs,
                         shuffle=True,
                         verbose=1,
                         callbacks=[lr_schedule])

    model.compile(optimizer=optimizer3,
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


if __name__ == '__main__':
    p = multiprocessing.Process(target=meteorSortLearningRate)
    p.start()
    p.join()
