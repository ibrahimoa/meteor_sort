from typing import Any

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from tensorflow.keras.models import Sequential


def get_performance_measures(model: Sequential, data_dir: str, image_resolution: (str, str), performance_file_path: str,
                             threshold: float = 0.5) -> None:
    """
    Evaluate the given model in the given dataset. Generate the confusion matrix and some performance measures
    (precision, recall and F1-Score) and save them in the given path.

    :param model: model to be evaluated
    :param data_dir: dataset where the model is evaluated
    :param image_resolution: image resolution to be used in the evaluation
    :param performance_file_path: file to store the results of the evaluation
    :param threshold: threshold to consider a result positive or negative (default is 0.5)
    :return:
    """
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)

    validation_generator = validation_datagen.flow_from_directory(data_dir,
                                                                  batch_size=1,
                                                                  class_mode='binary',
                                                                  color_mode='grayscale',
                                                                  target_size=image_resolution,
                                                                  shuffle=False)
    prob_predicted = model.predict(validation_generator, steps=len(validation_generator.filenames))
    validation_labels = []

    for i in range(0, len(validation_generator.filenames)):
        validation_labels.extend(np.array(validation_generator[i][1]))

    # Get the confusion matrix:
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_error: float = 0.0

    for i in range(len(prob_predicted)):
        if prob_predicted[i] >= threshold and validation_labels[i] == 1.0:
            true_positives += 1
        elif prob_predicted[i] >= threshold and validation_labels[i] == 0.0:
            false_positives += 1
        elif validation_labels[i] == 0.0:
            true_negatives += 1
        elif validation_labels[i] == 1.0:
            false_negatives += 1
        total_error += abs(prob_predicted[i] - validation_labels[i])

    if true_positives + true_negatives <= 0:
        raise Exception("Invalid confusion matrix generated. Please call the function with the correct parameters.")

    model_precision = true_positives / (true_positives + false_positives)
    model_recall = true_positives / (true_positives + false_negatives)
    if model_precision + model_recall <= 0:
        model_f1score = 0
    else:
        model_f1score = (2 * (model_precision * model_recall)) / (model_precision + model_recall)

    average_error: float = total_error / len(prob_predicted)

    with open(performance_file_path, 'w') as f:
        f.write('*********************************************\n')
        f.write('confusion matrix: \n')
        f.write('true positives: {}\n'.format(true_positives))
        f.write('false positives: {}\n'.format(false_positives))
        f.write('true negatives: {}\n'.format(true_negatives))
        f.write('false negatives: {}\n'.format(false_negatives))
        f.write('*********************************************\n')

        f.write('\n*********************************************\n')
        f.write('Performance metrics: \n')
        f.write('Model Precision: {}\n'.format(model_precision))
        f.write('Model Recall: {}\n'.format(model_recall))
        f.write('Model F1 Score: {}\n'.format(model_f1score))
        f.write('*********************************************\n')

        f.write('\n*********************************************\n')
        f.write('Model Average Error: {}\n'.format(average_error))
        f.write('*********************************************\n')

    return


def plot_acc_and_loss(history: Any, results_dir: str, model_number: str) -> None:
    """
    Plot the accuracy and loss from the history of a model and stores the images in the given path.

    :param history: history of the model (obtained from model.fit() function in Keras)
    :param results_dir: path where to store the plots images
    :param model_number: model number to be used in the images name
    :return:
    """
    try:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        # Get number of epochs
        epochs = range(len(acc))

        plt.plot(epochs, acc)
        plt.plot(epochs, val_acc)
        plt.title('Meteor detection training and validation Accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend(['Training', 'Validation'])
        plt.savefig(join(results_dir, 'results_' + model_number + '_acc'))

        plt.figure()
        plt.plot(epochs, loss)
        plt.plot(epochs, val_loss)
        plt.title('Meteor detection training and validation Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend(['Training', 'Validation'])
        plt.savefig(join(results_dir, 'results_' + model_number + '_loss'))
        plt.show()

    except Exception as e:
        print(e)
        raise ValueError(
            "Error in parameter 'history'. This parameter has to be the one returned from model.fit() function")
