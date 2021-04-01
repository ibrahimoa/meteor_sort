import tensorflow as tf
from tensorflow.keras.layers import Dense,  Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from os import listdir
import multiprocessing
from performanceMeasure import getPerformanceMeasures, plotAccuracyAndLoss

def trainCNN( ):

    tf.keras.backend.clear_session()

    modelNumber = 'model_2_07'
    base_dir = 'C:\work_dir\meteorData\extraData_70_30'
    results_dir = join('G:\GIEyA\TFG\meteor_classification\\results_2', modelNumber)
    results_dir_weights = join(results_dir, 'weights')

    train_dir = join(base_dir, 'train')
    validation_dir = join(base_dir, 'validation')

    ImageResolution = (432, 432)
    ImageResolutionGrayScale = (432, 432, 1)

    # Training -> 62483 (3905x16)
    # Validation -> 26780 (1673x16)

    training_images = len(listdir(join(train_dir, 'meteors'))) + len(listdir(join(train_dir, 'non_meteors')))
    validation_images = len(listdir(join(validation_dir, 'meteors'))) + len(listdir(join(validation_dir, 'non_meteors')))
    batch_size = 20
    steps_per_epoch = int(training_images / batch_size)
    validation_steps = int(validation_images / batch_size)

    #Rescale all images by 1./255

    train_datagen = ImageDataGenerator(rescale=1.0/255)

    validation_datagen = ImageDataGenerator(rescale=1.0/255.)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=batch_size,
                                                        class_mode='binary',
                                                        color_mode='grayscale',
                                                        target_size=ImageResolution)

    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  batch_size=batch_size,
                                                                  class_mode='binary',
                                                                  color_mode='grayscale',
                                                                  target_size=ImageResolution)

    # elu activation vs relu activation -> model_2_02 and model_2_03
    # dropout evaluation: model_2_02 (dropout .3) vs model_2_06 (no dropout) vs model_2_07 (dropout .4):

    model = tf.keras.models.Sequential([
        Conv2D(16, (7, 7), activation='elu', input_shape=ImageResolutionGrayScale, strides=1),
        Conv2D(16, (7, 7), activation='elu', input_shape=ImageResolutionGrayScale, strides=1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.40),

        Conv2D(12, (5, 5), activation='elu', kernel_initializer='he_uniform'),
        Conv2D(24, (5, 5), activation='elu', kernel_initializer='he_uniform'),
        Conv2D(12, (5, 5), activation='elu', kernel_initializer='he_uniform'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.40),

        Conv2D(12, (5, 5), activation='elu', kernel_initializer='he_uniform'),
        Conv2D(24, (5, 5), activation='elu', kernel_initializer='he_uniform'),
        Conv2D(12, (5, 5), activation='elu', kernel_initializer='he_uniform'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.40),

        Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_uniform'),
        Conv2D(24, (3, 3), activation='elu', kernel_initializer='he_uniform'),
        Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_uniform'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.40),

        Conv2D(24, (3, 3), activation='elu', kernel_initializer='he_uniform'),
        Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_uniform'),
        Conv2D(24, (3, 3), activation='elu', kernel_initializer='he_uniform'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.40),

        Flatten(),
        Dense(864, activation='elu', kernel_initializer='he_uniform'),
        Dropout(0.40),
        Dense(16, activation='elu', kernel_initializer='he_uniform'),
        Dropout(0.30),
        Dense(1, activation='sigmoid', kernel_initializer='he_uniform')
    ])

    print(model.summary())
    optimizer = Adam(learning_rate=5e-4)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    class SaveModelCallback(Callback):
        def __init__(self, thresholdTrain, thresholdValid):
            super(SaveModelCallback, self).__init__()
            self.thresholdTrain = thresholdTrain
            self.thresholdValid = thresholdValid

        def on_epoch_end(self, epoch, logs=None):
            if((logs.get('accuracy') >= self.thresholdTrain) and (logs.get('val_accuracy') >= self.thresholdValid)):
                model.save_weights(join(results_dir_weights, modelNumber + '_acc_' +  str(logs.get('accuracy'))[0:5]
                                        + '_val_acc_' + str(logs.get('val_accuracy'))[0:5] + '.h5'), save_format='h5')

    callback_84_84 = SaveModelCallback(0.840, 0.840)

    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=15, #Later train with more epochs if neccessary
                        validation_steps=validation_steps,
                        shuffle=True,
                        verbose=1,
                        callbacks=[callback_84_84])

    ################################# PRINT MODEL PERFORMANCE AND GET PERFORMANCE MEASURES  #################################

    # Get performance measures:
    getPerformanceMeasures(model, validation_dir, ImageResolution, join(results_dir, 'performance_' + modelNumber + '.txt'), threshold=0.50)

    # Plot Accuracy and Loss in both train and validation sets
    plotAccuracyAndLoss(history)

    #########################################################################################################################

if __name__ == '__main__':
    p = multiprocessing.Process(target=trainCNN)
    p.start()
    p.join()

