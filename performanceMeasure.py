import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join


def getPerformanceMeasures(model, dataDir, ImageResolution, performanceFile, threshold=0.5) -> None:
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)

    validation_generator = validation_datagen.flow_from_directory(dataDir,
                                                                  batch_size=1,
                                                                  class_mode='binary',
                                                                  color_mode='grayscale',
                                                                  target_size=ImageResolution,
                                                                  shuffle=False)
    prob_predicted = model.predict(validation_generator, steps=len(validation_generator.filenames))
    validation_labels = []

    for i in range(0, len(validation_generator.filenames)):
        validation_labels.extend(np.array(validation_generator[i][1]))

    #############################################################################################

    # Get the confusion matrix:
    truePositives: int = 0
    trueNegatives: int = 0
    falsePositives: int = 0
    falseNegatives: int = 0
    totalError: float = 0.0

    for i in range(len(prob_predicted)):
        if (prob_predicted[i] >= threshold and validation_labels[i] == 1.0):
            truePositives += 1
        elif (prob_predicted[i] >= threshold and validation_labels[i] == 0.0):
            falsePositives += 1
        elif (validation_labels[i] == 0.0):
            trueNegatives += 1
        elif (validation_labels[i] == 1.0):
            falseNegatives += 1
        totalError += abs(prob_predicted[i] - validation_labels[i])
    try:
        modelPrecision = (truePositives) / (truePositives + falsePositives)
    except:
        modelPrecision = 0

    try:
        modelRecall = (truePositives) / (truePositives + falseNegatives)
    except:
        modelRecall = 0

    try:
        modelF1score = (2 * (modelPrecision * modelRecall)) / (modelPrecision + modelRecall)
    except:
        modelF1score = 0

    averageError: float = totalError / len(prob_predicted)

    with open(performanceFile, 'w') as performanceFile:
        performanceFile.write('*********************************************\n')
        performanceFile.write('confusion matrix: \n')
        performanceFile.write('true positives: {}\n'.format(truePositives))
        performanceFile.write('false positives: {}\n'.format(falsePositives))
        performanceFile.write('true negatives: {}\n'.format(trueNegatives))
        performanceFile.write('false negatives: {}\n'.format(falseNegatives))
        performanceFile.write('*********************************************\n')

        performanceFile.write('\n*********************************************\n')
        performanceFile.write('Performance metrics: \n')
        performanceFile.write('Model Precision: {}\n'.format(modelPrecision))
        performanceFile.write('Model Recall: {}\n'.format(modelRecall))
        performanceFile.write('Model F1 Score: {}\n'.format(modelF1score))
        performanceFile.write('*********************************************\n')

        performanceFile.write('\n*********************************************\n')
        performanceFile.write('Model Average Error: {}\n'.format(averageError))
        performanceFile.write('*********************************************\n')

    #############################################################################################
    return


def plotAccuracyAndLoss(history, results_dir: str, model_number: str) -> None:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))  # Get number of epochs

    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Precisión de validación y entrenamiento en la detección de meteoros')  # Meteor detection training and validation accuracy
    plt.xlabel('Iteraciones')
    plt.ylabel('Precisión')
    plt.savefig(join(results_dir, 'results' + model_number + '_acc'))

    plt.figure()
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Error de validación y entrenamiento en la detección de meteoros')  # Meteor detection training and validation loss
    plt.xlabel('Iteraciones')
    plt.ylabel('Error')
    plt.savefig(join(results_dir, 'results' + model_number + '_loss'))
    plt.show()

    return


def getProblematicMeteors(model, dataDir, ImageResolution, problematicMeteorsFile, margin=0.40) -> None:
    datagen = ImageDataGenerator(rescale=1.0 / 255.)

    generator = datagen.flow_from_directory(dataDir,
                                            batch_size=1,
                                            class_mode='binary',
                                            color_mode='grayscale',
                                            target_size=ImageResolution,
                                            shuffle=False)
    prob_predicted = model.predict(generator, steps=len(generator.filenames))
    labels = []

    for i in range(0, len(generator.filenames)):
        labels.extend(np.array(generator[i][1]))

    #############################################################################################

    problematicPositives: int = 0
    problematicNegatives: int = 0
    totalError: float = 0.0

    with open(problematicMeteorsFile, 'w') as problematicFile:
        for i in range(len(prob_predicted)):
            if (labels[i] == 1.0):
                totalError += labels[i] - prob_predicted[i]
                if (prob_predicted[i] <= (0.5 - margin)):
                    problematicFile.write('meteors_{}\n'.format(i))
                    problematicPositives += 1
            else:
                totalError += prob_predicted[i]
                if (prob_predicted[i] >= (0.5 + margin)):
                    problematicFile.write('non_meteors_{}\n'.format(i))
                    problematicNegatives += 1

    relativeError: float = totalError / len(prob_predicted)
    print('\n\n\n\n*********************************************')
    print('Margin: {}'.format(margin))
    print('Problematic meteors: {}'.format(problematicPositives))
    print('Problematic non-meteors: {}'.format(problematicNegatives))
    print('Relative error: {}'.format(relativeError))
    print('*********************************************\n\n\n\n')

    # Margin: 0.3
    # Problematic meteors: 2729
    # Problematic non-meteors: 1263

    # Margin: 0.35
    # Problematic meteors: 2410
    # Problematic non-meteors: 969

    # Margin: 0.4
    # Problematic meteors: 2016
    # Problematic non-meteors: 650

    # Margin: 0.45
    # Problematic meteors: 1344
    # Problematic non-meteors: 326

    # Margin: 0.48
    # Problematic meteors: 730
    # Problematic non-meteors: 134

    #############################################################################################

    # Margin: 0.4
    # Problematic meteors: 1200
    # Problematic non-meteors: 561
    # Relative error: [0.13366854]

    return
