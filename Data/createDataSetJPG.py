import os
from os.path import join
import astropy.io.fits as ft
import numpy as np
import imageio

dataDir = 'D:\MeteorDataBase\ServerData'
truePos = 'truePositivesMeteors.txt'
falsePos = 'falsePositivesMeteors.txt'

finalDir = 'C:\work_dir\meteor_classification\labeledData'
meteorsFinalDir = join(finalDir, 'meteors')
nonMeteorsFinalDir = join(finalDir, 'non_meteors')

be0001BaseDir = 'D:\MeteorDataBase\ServerData\\be0001\\files\processed'
be0002BaseDir = 'D:\MeteorDataBase\ServerData\\be0002\\files\processed'
be0003BaseDir = 'D:\MeteorDataBase\ServerData\\be0003\\files\processed'
be0004BaseDir = 'D:\MeteorDataBase\ServerData\\be0004\\files\processed'
de0001BaseDir = 'D:\MeteorDataBase\ServerData\\de0001\\files\processed'
nl0009BaseDir = 'D:\MeteorDataBase\ServerData\\nl0009\\files\processed'


tp = open(join(dataDir, truePos), 'r')
fp = open(join(dataDir, falsePos), 'r')
tpList = [line.rstrip('\r\n') for line in tp]
fpList = [line.rstrip('\r\n') for line in fp]
print('There are ' + str(len(tpList)) + ' meteors (true positives)') #It should be 25402
print('There are ' + str(len(fpList)) + ' non-meteors (false positives)') #It should be 44750

totalMeteors = 0
totalNonMeteors = 0

for root, dirs, files in os.walk(be0001BaseDir):
    for file in files:
        if(file in tpList):
            fitsFile = ft.open(join(root,file))
            imageio.imwrite(join(meteorsFinalDir, 'meteors_' + str(totalMeteors) + '.jpg'), fitsFile[1].data)
            totalMeteors += 1
        elif(file in fpList):
            fitsFile = ft.open(join(root,file))
            imageio.imwrite(join(nonMeteorsFinalDir, 'non_meteors_' + str(totalNonMeteors) + '.jpg'), fitsFile[1].data)
            totalNonMeteors +=1
print(str(totalMeteors) + ' meteors were found in BE0001')
print(str(totalNonMeteors) + ' non-meteors were found in BE0001')
print('----------------------------------------')

for root, dirs, files in os.walk(be0002BaseDir):
    for file in files:
        if(file in tpList):
            fitsFile = ft.open(join(root,file))
            imageio.imwrite(join(meteorsFinalDir, 'meteors_' + str(totalMeteors) + '.jpg'), fitsFile[1].data)
            totalMeteors += 1
        elif(file in fpList):
            fitsFile = ft.open(join(root,file))
            imageio.imwrite(join(nonMeteorsFinalDir, 'non_meteors_' + str(totalNonMeteors) + '.jpg'), fitsFile[1].data)
            totalNonMeteors +=1
print(str(totalMeteors) + ' meteors were found in BE0002')
print(str(totalNonMeteors) + ' non-meteors were found in BE0002')
print('----------------------------------------')

for root, dirs, files in os.walk(be0003BaseDir):
    for file in files:
        if(file in tpList):
            fitsFile = ft.open(join(root,file))
            imageio.imwrite(join(meteorsFinalDir, 'meteors_' + str(totalMeteors) + '.jpg'), fitsFile[1].data)
            totalMeteors += 1
        elif(file in fpList):
            fitsFile = ft.open(join(root,file))
            imageio.imwrite(join(nonMeteorsFinalDir, 'non_meteors_' + str(totalNonMeteors) + '.jpg'), fitsFile[1].data)
            totalNonMeteors +=1
print(str(totalMeteors) + ' meteors were found in BE0003')
print(str(totalNonMeteors) + ' non-meteors were found in BE0003')
print('----------------------------------------')

for root, dirs, files in os.walk(be0004BaseDir):
    for file in files:
        if(file in tpList):
            fitsFile = ft.open(join(root,file))
            imageio.imwrite(join(meteorsFinalDir, 'meteors_' + str(totalMeteors) + '.jpg'), fitsFile[1].data)
            totalMeteors += 1
        elif(file in fpList):
            fitsFile = ft.open(join(root,file))
            imageio.imwrite(join(nonMeteorsFinalDir, 'non_meteors_' + str(totalNonMeteors) + '.jpg'), fitsFile[1].data)
            totalNonMeteors +=1
print(str(totalMeteors) + ' meteors were found in BE0004')
print(str(totalNonMeteors) + ' non-meteors were found in BE0004')
print('----------------------------------------')

for root, dirs, files in os.walk(de0001BaseDir):
    for file in files:
        if(file in tpList):
            fitsFile = ft.open(join(root,file))
            imageio.imwrite(join(meteorsFinalDir, 'meteors_' + str(totalMeteors) + '.jpg'), fitsFile[1].data)
            totalMeteors += 1
        elif(file in fpList):
            fitsFile = ft.open(join(root,file))
            imageio.imwrite(join(nonMeteorsFinalDir, 'non_meteors_' + str(totalNonMeteors) + '.jpg'), fitsFile[1].data)
            totalNonMeteors +=1
print(str(totalMeteors) + ' meteors were found in DE0001')
print(str(totalNonMeteors) + ' non-meteors were found in DE0001')
print('----------------------------------------')
for root, dirs, files in os.walk(nl0009BaseDir):
    for file in files:
        if(file in tpList):
            fitsFile = ft.open(join(root,file))
            imageio.imwrite(join(meteorsFinalDir, 'meteors_' + str(totalMeteors) + '.jpg'), fitsFile[1].data)
            totalMeteors += 1
        elif(file in fpList):
            fitsFile = ft.open(join(root,file))
            imageio.imwrite(join(nonMeteorsFinalDir, 'non_meteors_' + str(totalNonMeteors) + '.jpg'), fitsFile[1].data)
            totalNonMeteors +=1 
print(str(totalMeteors) + ' meteors were found in NL0009')
print(str(totalNonMeteors) + ' non-meteors were found in NL0009')
print('----------------------------------------')





