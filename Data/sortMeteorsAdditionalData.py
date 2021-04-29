#@Author:       Ibrahim Oulad Amar
#@Organization: Universidad Politécnica de Madrid
#@Purpose:      Create two .txt files. The first one will contain all .fits files with meteors (manually filtered).
#               The second one will contain false positives (.fits files that are not real meteors)

#The Number Of True positives Is:  48473 (Meteors)
#The Number Of False positives Is:  65269 (Non-meteors)


import os


foldersDir = 'D:\MeteorDataBase\ServerData\AdditionalData\mldataset\\files\ConfirmedFiles'
truePos = 'truePositivesMeteorsAdditional.txt'
falsePos = 'falsePositivesMeteorsAdditional.txt'
tpList = []
fpList = []
allList = []
examinedFiles = 0


tp = open(os.path.join(foldersDir, truePos), 'w+')
fp = open(os.path.join(foldersDir, falsePos), 'w+')


for root, dirs, files in os.walk(foldersDir):
    for dir in dirs:
        for file in os.listdir(os.path.join(foldersDir, dir)):
            if(file.endswith('.txt')):
                f = open(os.path.join(foldersDir, dir, file), 'r')
                #'_pre-confirmation.txt' is the original filter. In order to optimize we only use 'n.txt'
                #since the other files end with a number.txt
                if(file.endswith('_uncalibrated_pre-confirmation.txt')):
                    pass
                elif(file.startswith('FTPdetectinfo_')):
                    if(file.endswith('_pre-confirmation.txt')):
                        # All detections
                        lines = [line.rstrip('\n') for line in f]
                        for line in lines:
                            if(line.endswith('.fits') and (line not in allList)):
                                allList.append(line)
                    else:
                        # Only manually filtered detections
                        lines = [line.rstrip('\n') for line in f]
                        for line in lines:
                            if(line.endswith('.fits') and (line not in tpList)):
                                tpList.append(line)
                    f.close()
                    examinedFiles += 1
                    if ((examinedFiles % 25) == 0):
                        print(str(examinedFiles) + ' files have been examined')


#False Positives are all FFs minus True Positives
fpList = [i for i in allList if i not in tpList]
print('The Number Of True positives Is: ', len(tpList))
print('The Number Of False positives Is: ', len(fpList))

#Now that we have the lists with all the files sorted we can write them on the .txt files:
for i in tpList:
    tp.write(i + '\r')

for j in fpList:
    fp.write(j + '\r')

#Close the files
tp.close()
fp.close()