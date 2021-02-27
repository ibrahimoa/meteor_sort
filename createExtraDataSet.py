import os
from os.path import join
import astropy.io.fits as ft
import numpy as np
import imageio

dataDir = 'D:\MeteorDataBase\ServerData\AdditionalData\mldataset\\files\ConfirmedFiles'
labelsDir = 'D:\MeteorDataBase\ServerData'
tp_new = 'new_meteors.txt'
fp_new = 'new_non_meteors.txt'

finalDir = 'C:\work_dir\meteorData\extraData'
meteorsFinalDir = join(finalDir, 'meteors')
nonMeteorsFinalDir = join(finalDir, 'non_meteors')

tp = open(join(labelsDir, tp_new), 'r')
fp = open(join(labelsDir, fp_new), 'r')
tpList = [line.rstrip('\r\n') for line in tp]
fpList = [line.rstrip('\r\n') for line in fp]
print('There are ' + str(len(tpList)) + ' new meteors (true positives)') #It should be 23068
print('There are ' + str(len(fpList)) + ' new non-meteors (false positives)') #It should be 20522
print('----------------------------------------')
totalMeteors = 24676
totalNonMeteors = 41674

for root, dirs, files in os.walk(dataDir):
    for dir in dirs:
        for file in os.listdir(os.path.join(dataDir, dir)):
            if(file.endswith('.fits')):
                if(file in tpList):
                    fitsFile = ft.open(join(dataDir, dir, file))
                    imageio.imwrite(join(meteorsFinalDir, 'meteors_' + str(totalMeteors) + '.jpg'), fitsFile[1].data)
                    totalMeteors += 1
                elif(file in fpList):
                    fitsFile = ft.open(join(dataDir, dir, file))
                    imageio.imwrite(join(nonMeteorsFinalDir, 'non_meteors_' + str(totalNonMeteors) + '.jpg'), fitsFile[1].data)
                    totalNonMeteors +=1
            else:
                pass

print(str(totalMeteors) + ' meteors in total')
print(str(totalNonMeteors) + ' non-meteors in total')
print('----------------------------------------')







