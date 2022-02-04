from PIL import Image
from os.path import join
from os import walk


def resizeFiles(origin, destiny):
    for root, dirs, files in walk(origin):
        for file in files:
            image = Image.open(join(root, file), 'r')
            img_res = image.resize((640, 360))
            img_res.save(join(destiny, file))


train_orig = 'G:\GIEyA\TFG\meteor_classification\labeledData\\train'
validation_orig = 'G:\GIEyA\TFG\meteor_classification\labeledData\\validation'
train_resize = 'G:\GIEyA\TFG\meteor_classification\labeledData\\train_640x360'
validation_resize = 'G:\GIEyA\TFG\meteor_classification\labeledData\\validation_640x360'

train_met = join(train_orig, 'meteors')
train_non_met = join(train_orig, 'non_meteors')
valid_met = join(validation_orig, 'meteors')
valid_non_met = join(validation_orig, 'non_meteors')
tr_met = join(train_resize, 'meteors')
tr_non_met = join(train_resize, 'non_meteors')
vr_met = join(validation_resize, 'meteors')
vr_non_met = join(validation_resize, 'non_meteors')

# resizeFiles(train_met, tr_met)
resizeFiles(train_non_met, tr_non_met)
# resizeFiles(valid_met, vr_met)
# resizeFiles(valid_non_met, vr_non_met)
