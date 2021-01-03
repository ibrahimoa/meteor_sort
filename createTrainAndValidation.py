import os
import random
from shutil import copyfile


# Write a python function called split_data which takes
# a SOURCE directory containing the files
# a TRAINING directory that a portion of the files will be copied to
# a TESTING directory that a portion of the files will be copie to
# a SPLIT SIZE to determine the portion
# The files should also be randomized, so that the training set is a random
# X% of the files, and the test set is the remaining files
# SO, for example, if SOURCE is PetImages/Cat, and SPLIT SIZE is .9
# Then 90% of the images in PetImages/Cat will be copied to the TRAINING dir
# and 10% of the images will be copied to the TESTING dir
# Also -- All images should be checked, and if they have a zero file length,
# they will not be copied over

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    # YOUR CODE STARTS HERE
    contentList = os.listdir(SOURCE)
    for i in range(len(contentList)):
        if (os.path.getsize(os.path.join(SOURCE, contentList[i])) == 0):
            contentList.pop(i)
    random.sample(contentList, len(contentList))
    for i in range(int(SPLIT_SIZE * len(contentList))):
        copyfile(os.path.join(SOURCE, contentList[i]), os.path.join(TRAINING, contentList[i]))
    for i in range(int(SPLIT_SIZE * len(contentList)), len(contentList)):
        copyfile(os.path.join(SOURCE, contentList[i]), os.path.join(TESTING, contentList[i]))


# YOUR CODE ENDS HERE


METEORS_SOURCE_DIR = "C:\work_dir\MeteorClassificationProject\labeledData\meteors"
TRAINING_METEORS_DIR = "C:\work_dir\MeteorClassificationProject\labeledData\\train\meteors"
TESTING_METEORS_DIR = "C:\work_dir\MeteorClassificationProject\labeledData\\validation\meteors"
NON_METEORS_SOURCE_DIR = "C:\work_dir\MeteorClassificationProject\labeledData\\non_meteors"
TRAINING_NON_METEORS_DIR = "C:\work_dir\MeteorClassificationProject\labeledData\\train\\non_meteors"
TESTING_NON_METEORS_DIR = "C:\work_dir\MeteorClassificationProject\labeledData\\validation\\non_meteors"

if __name__ == "__main__":
    split_size = .8
    split_data(METEORS_SOURCE_DIR, TRAINING_METEORS_DIR, TESTING_METEORS_DIR, split_size)
    split_data(NON_METEORS_SOURCE_DIR, TRAINING_NON_METEORS_DIR, TESTING_NON_METEORS_DIR, split_size)