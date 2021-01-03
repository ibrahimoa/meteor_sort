import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import multiprocessing

print("My current tensorflow version is: ", tf.__version__)

def trainCNN( ):

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
                                       rotation_range=10, #Range from 0 to 180 degrees to randomly rotate images. In this case it's going to rotate between 0 and 40 degrees
                                       width_shift_range=0.05, #Move image in this fram (20%)
                                       height_shift_range=0.05,
                                       #shear_range=0.2, #Girar la imagen un 20%
                                       #zoom_range=0.5, #Zoom up-to 20%
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

    model.compile(optimizer=Adam(learning_rate=0.001), #RMSprop(lr=0.001)
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    #39.480 -> Training
    #9.872 -> Validation

    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=1233, #4935
                        epochs=30, #Later train with more epochs if neccessary
                        validation_steps=308, #1234
                        verbose=1)

    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc)) #Get number of epochs

    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Meteor detection training and validation accuracy')
    plt.figure()

    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Meteor detection training and validation loss')

    plt.show()

if __name__ == '__main__':
    p = multiprocessing.Process(target=trainCNN)
    p.start()
    p.join()