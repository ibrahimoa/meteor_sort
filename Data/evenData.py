import os
import random
from shutil import copyfile
from os.path import join


def evenData(ORIGIN, DESTINY):
    print(join(ORIGIN, 'meteors'))
    n = len(os.listdir(join(ORIGIN, 'meteors')))
    meteors = os.listdir(join(ORIGIN, 'meteors'))
    nonMeteors = os.listdir(join(ORIGIN, 'non_meteors'))
    for i in range(n):
        copyfile(join(ORIGIN, join('meteors', meteors[i])), join(DESTINY, join('meteors', meteors[i])))
        copyfile(join(ORIGIN, join('non_meteors', nonMeteors[i])), join(DESTINY, join('non_meteors', nonMeteors[i])))


ORIGIN_DATA = 'G:\GIEyA\TFG\meteor_classification\labeledData'
ORIGIN_TRAIN = join(ORIGIN_DATA, 'train')
ORIGIN_VALID = join(ORIGIN_DATA, 'validation')

EVEN_DATA = 'G:\GIEyA\TFG\meteor_classification\labeledData\evenData'
EVEN_TRAIN = join(EVEN_DATA, 'train')
EVEN_VALID = join(EVEN_DATA, 'valid')

if __name__ == "__main__":
    evenData(ORIGIN_TRAIN, EVEN_TRAIN)
    evenData(ORIGIN_VALID, EVEN_VALID)
