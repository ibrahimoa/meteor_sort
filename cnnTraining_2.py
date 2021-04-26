import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from os.path import join
from os import listdir
import multiprocessing
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import unit_norm
from performanceMeasure import getPerformanceMeasures, plotAccuracyAndLoss, getProblematicMeteors


def trainCNN():
    tf.keras.backend.clear_session()

    modelNumber = 'model_2_21'
    base_dir = 'C:\work_dir\meteorData\extraData_85_15'  # We don't use filtered data ... Not so useful
    results_dir = join('G:\GIEyA\TFG\meteor_classification\\results_2', modelNumber)
    results_dir_weights = join(results_dir, 'weights')

    train_dir = join(base_dir, 'train')
    validation_dir = join(base_dir, 'validation')

    ImageResolution: tuple = (256, 256)  # (432, 432) | (300, 300) |
    ImageResolutionGrayScale: tuple = (256, 256, 1)  # (432, 432, 1) | (300, 300, 1)
    DROPOUT: float = 0.30
    EPOCHS: int = 200
    LEARNING_RATE: float = 5e-4

    training_images = len(listdir(join(train_dir, 'meteors'))) + len(listdir(join(train_dir, 'non_meteors')))
    validation_images = len(listdir(join(validation_dir, 'meteors'))) + len(listdir(join(validation_dir, 'non_meteors')))
    batch_size: int = 64
    steps_per_epoch: int = int(training_images / batch_size)
    validation_steps: int = int(validation_images / batch_size)

    # Rescale all images by 1./255

    train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                       rotation_range=10,  # Range from 0 to 180 degrees to randomly rotate images
                                       width_shift_range=0.05,
                                       height_shift_range=0.05,
                                       shear_range=5,  # Shear the image by 5 degrees
                                       zoom_range=0.1,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode='nearest'
                                       )

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)

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
    # dropout evaluation: model_2_02 (.3) vs model_2_06 (no dropout) vs model_2_07 (.4) vs model_2_08 (.5):
    # model 2.9 -> Simple CNN (5 conv layers + 2 fully-connected) -> Only 123,209 parameters. Training time: 550 s/epoch
    # model 2.10 -> 2.9 with filtered data
    # model 2.11 -> Very complex CNN + BatchNormalization (???) -> ??? parameters. Training time: ???
    # model 2.12 -> Add regularization and weight constrains : Not so useful (discarded)
    # [kernel_regularizer=l2(l=0.01) + kernel_constraint=unit_norm() + BatchNormalization()]
    # new model 2.12 -> BatchNormalization + kernel_regularizer
    # model 2.13 -> BatchNormalization + unit_norm()
    # model 2.14 -> Make it simpler in order to avoid overfitting
    # model 2.15 -> Simpler and smaller input size
    # model 2.16 -> Simpler
    # model 2.17 -> Smaller image size (just to compare it with the previous one)
    # model 2.18 -> Split data in 0.85 and 0.15 and simpler (4 convolutions vs 5)
    # model 2.19 -> Model 2.18 + Data Augmentation
    # model 2.20 -> Try with more data augmentation (horizontal_flip and vertical_flip)
    # model 2.21 -> Model 2.20 with more epochs

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
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    class SaveModelCallback(Callback):
        def __init__(self, thresholdTrain, thresholdValid):
            super(SaveModelCallback, self).__init__()
            self.thresholdTrain = thresholdTrain
            self.thresholdValid = thresholdValid

        def on_epoch_end(self, epoch, logs=None):
            if ((logs.get('accuracy') >= self.thresholdTrain) and (logs.get('val_accuracy') >= self.thresholdValid)):
                model.save_weights(join(results_dir_weights, modelNumber + '_acc_' + str(logs.get('accuracy'))[0:5]
                                        + '_val_acc_' + str(logs.get('val_accuracy'))[0:5] + '.h5'), save_format='h5')

    callback_92_92 = SaveModelCallback(0.92, 0.92)

    history = model.fit(train_generator,
                       validation_data=validation_generator,
                       steps_per_epoch=steps_per_epoch,
                       epochs=EPOCHS,
                       validation_steps=validation_steps,
                       shuffle=True,
                       verbose=2,
                       callbacks=[callback_92_92])

    ################################# PRINT MODEL PERFORMANCE AND GET PERFORMANCE MEASURES  #################################

    # Load best model weights:
    #model.load_weights(join(results_dir_weights, 'model_2_21_acc_0.944_val_acc_0.939.h5'))

    # Get performance measures:
    getPerformanceMeasures(model, train_dir, ImageResolution,
                           join(results_dir, 'performance_' + modelNumber + '.txt'), threshold=0.50)

    # Plot Accuracy and Loss in both train and validation sets
    plotAccuracyAndLoss(history, results_dir, modelNumber[-5:])

    #########################################################################################################################


if __name__ == '__main__':
    p = multiprocessing.Process(target=trainCNN)
    p.start()
    p.join()
