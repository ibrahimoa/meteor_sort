import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

def getPerformanceMeasures(model, dataDir, ImageResolution, performanceFile, threshold=0.5):
    validation_datagen = ImageDataGenerator(rescale=1.0/255.)

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
    truePositives = 0
    trueNegatives = 0
    falsePositives = 0
    falseNegatives = 0

    for i in range(len(prob_predicted)):
        if(prob_predicted[i] >= threshold and validation_labels[i] == 1.0):
            truePositives += 1
        elif(prob_predicted[i] >= threshold and validation_labels[i] == 0.0):
            falsePositives += 1
        elif(validation_labels[i] == 0.0):
            trueNegatives += 1
        elif(validation_labels[i] == 1.0):
            falseNegatives += 1

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

    with open(performanceFile, 'w') as performanceFile:
        performanceFile.write('*********************************************\n')
        performanceFile.write('confusion matrix: \n')
        performanceFile.write('true positives: {}\n'.format(truePositives))
        performanceFile.write('false positives: {}\n'.format(falsePositives))
        performanceFile.write('true negatives: {}\n'.format(trueNegatives))
        performanceFile.write('false negatives: {}\n'.format(falseNegatives))
        performanceFile.write('*********************************************\n')


        performanceFile.write('*********************************************\n')
        performanceFile.write('Performance metrics: \n')
        performanceFile.write('Model Precision: {}\n'.format(modelPrecision))
        performanceFile.write('Model Recall: {}\n'.format(modelRecall))
        performanceFile.write('Model F1 Score: {}\n'.format(modelF1score))
        performanceFile.write('*********************************************\n')

    #############################################################################################
    return

def plotAccuracyAndLoss(history):

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

    return



