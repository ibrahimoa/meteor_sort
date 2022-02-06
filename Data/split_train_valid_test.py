import os
import random
from shutil import copyfile
from pathlib import Path
from os import getcwd
from os.path import join


def split_data(source_dir: str, training_dir: str, validation_dir: str, testing_dir: str, split_training: float,
               split_validation: float, split_testing: float) -> None:
    """
    This function splits a given dataset into three sets: training, validation and testing.

    :param source_dir: the dataset directory path
    :param training_dir: the training directory path
    :param validation_dir: the validation directory path
    :param testing_dir: the testing directory path
    :param split_training: the training split
    :param split_validation: the validation split
    :param split_testing: the testing split
    :return: None

    Note: the sum of the three arguments `split_training`, `split_validation` and `split_testing` has to be equal to 1.
    """
    # Check the values of 'split_training', 'split_validation' and 'split_testing'.
    if split_training + split_validation + split_testing != 1:
        raise ValueError("The sum of 'split_training', 'split_validation' and 'split_testing' arguments has to be "
                         "equal to 1")
    content_list = os.listdir(source_dir)

    random.sample(content_list, len(content_list))

    # Training set.
    for i in range(int(split_training * len(content_list))):
        try:
            copyfile(os.path.join(source_dir, content_list[i]), os.path.join(training_dir, content_list[i]))
        except:
            raise Exception(
                "Couldn't copy the file '{}' into the location '{}'".format(os.path.join(source_dir, content_list[i]),
                                                                            os.path.join(training_dir,
                                                                                         content_list[i])))

    # Validation set.
    for i in range(int(split_training * len(content_list)),
                   int((split_training + split_validation) * len(content_list))):
        try:
            copyfile(os.path.join(source_dir, content_list[i]), os.path.join(validation_dir, content_list[i]))
        except:
            raise Exception(
                "Couldn't copy the file '{}' into the location '{}'".format(os.path.join(source_dir, content_list[i]),
                                                                            os.path.join(validation_dir,
                                                                                         content_list[i])))

    # Testing set.
    for i in range(int((split_training + split_validation) * len(content_list)), len(content_list)):
        try:
            copyfile(os.path.join(source_dir, content_list[i]), os.path.join(testing_dir, content_list[i]))
        except:
            raise Exception(
                "Couldn't copy the file '{}' into the location '{}'".format(os.path.join(source_dir, content_list[i]),
                                                                            os.path.join(testing_dir,
                                                                                         content_list[i])))


data_dir = join(Path(getcwd()).parent, "meteor_data")
meteors_source_dir = join(data_dir, "meteors")
non_meteors_source_dir = join(data_dir, "non_meteors")

training_set_meteors_dir = join(data_dir, "train/meteors")
validation_set_meteors_dir = join(data_dir, "validation/meteors")
testing_set_meteors_dir = join(data_dir, "test/meteors")

training_set_non_meteors_dir = join(data_dir, "train/non_meteors")
validation_set_non_meteors_dir = join(data_dir, "validation/non_meteors")
testing_set_non_meteors_dir = join(data_dir, "test/non_meteors")

if __name__ == "__main__":
    split_data(meteors_source_dir, training_set_meteors_dir, validation_set_meteors_dir, testing_set_meteors_dir, 0.85,
               0.15, 0.00)
    split_data(non_meteors_source_dir, training_set_non_meteors_dir, validation_set_non_meteors_dir,
               testing_set_non_meteors_dir, 0.90,
               0.10, 0.00)
