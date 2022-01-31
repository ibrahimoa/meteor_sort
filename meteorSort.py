import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from os.path import join
from os import listdir, getcwd
import multiprocessing
from tensorflow.keras.constraints import unit_norm
from performanceMeasure import getPerformanceMeasures, plotAccuracyAndLoss
from tensorflow import lite
from meteorSortLearningRate import meteorSortLearningRate


class MeteorSortCallback(Callback):
    def __init__(self, threshold_train, threshold_valid, model, model_description, dir_weights):
        super(MeteorSortCallback, self).__init__()
        self.thresholdTrain = threshold_train
        self.thresholdValid = threshold_valid
        self.model_description = model_description
        self.dir_weights = dir_weights
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('accuracy') >= self.threshold_train) and (logs.get('val_accuracy') >= self.threshold_valid):
            self.model.save_weights(
                join(self.dir_weights, self.model_description + '_acc_' + str(logs.get('accuracy'))[0:5]
                     + '_val_acc_' + str(logs.get('val_accuracy'))[0:5] + '.h5'), save_format='h5')


def meteorSort():
    tf.keras.backend.clear_session()

    # Data
    data_dir = join(getcwd(), "meteorData")
    train_dir = join(data_dir, 'train')
    validation_dir = join(data_dir, 'validation')

    # Model handling
    model_to_convert = ""
    model_name = 'final_model'
    results_dir = join('.\\Results', model_name)
    results_dir_weights = join(results_dir, 'weights')

    # Hyperparameters for the training
    image_resolution: tuple = (256, 256)
    image_resolution_gray_scale: tuple = (256, 256, 1)
    epochs: int = 200
    learning_rate: float = 5e-4
    get_ideal_learning_rate: bool = False
    train_set_threshold: float = 0.92
    validation_set_threshold: float = 0.92

    num_training_images = len(listdir(join(train_dir, 'meteors'))) + len(listdir(join(train_dir, 'non_meteors')))
    num_validation_images = len(listdir(join(validation_dir, 'meteors'))) \
                            + len(listdir(join(validation_dir, 'non_meteors')))
    batch_size: int = 64
    steps_per_epoch: int = int(num_training_images / batch_size)
    validation_steps: int = int(num_validation_images / batch_size)

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
                                                        target_size=image_resolution)

    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  batch_size=batch_size,
                                                                  class_mode='binary',
                                                                  color_mode='grayscale',
                                                                  target_size=image_resolution)

    model = tf.keras.models.Sequential([

        Conv2D(8, (7, 7), activation='elu', input_shape=image_resolution_gray_scale,
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

    if get_ideal_learning_rate:
        meteorSortLearningRate(model, train_dir, image_resolution, batch_size, epochs, steps_per_epoch)

    print(model.summary())
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    callback_92_92 = MeteorSortCallback(train_set_threshold, validation_set_threshold, model, model_name,
                                        results_dir_weights)

    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_steps=validation_steps,
                        shuffle=True,
                        verbose=1,
                        callbacks=[callback_92_92])

    # ------ Print model performance and get performance measures  ------

    if model_to_convert != "":
        # Load best model weights:
        model.load_weights(join(results_dir, model_to_convert))

        # Convert model to tflite:
        converter = lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        open("meteorLiteModel.tflite", "wb").write(tflite_model)

        # Get performance measures:
        getPerformanceMeasures(model, train_dir, image_resolution,
                               join(results_dir, 'performance_' + model_name + '.txt'), threshold=0.50)

        # Plot Accuracy and Loss in both train and validation sets
        plotAccuracyAndLoss(history, results_dir, model_name[-5:])


if __name__ == '__main__':
    p = multiprocessing.Process(target=meteorSort)
    p.start()
    p.join()
