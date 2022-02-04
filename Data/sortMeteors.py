# @Author:       Ibrahim Oulad Amar
# @Organization: Universidad Politécnica de Madrid
# @Purpose:      Create two .txt files. The first one will contain all .fits files with meteors (manually filtered).
#               The second one will contain false positives (.fits files that are not real meteors)

# The Number Of True positives Is:  25402
# The Number Of False positives Is:  44750


import os

dataDir = 'D:\MeteorDataBase\ServerData'
truePos = 'truePositivesMeteors.txt'
falsePos = 'falsePositivesMeteors.txt'
labelsDir = os.path.join(dataDir, 'labels')
tpList = []
fpList = []
allList = []
examinedFiles = 0

tp = open(os.path.join(dataDir, truePos), 'w+')
fp = open(os.path.join(dataDir, falsePos), 'w+')

for root, dirs, files in os.walk(labelsDir):
    for file in files:
        f = open(os.path.join(labelsDir, file), 'r')
        # '_pre-confirmation.txt' is the original filter. In order to optimize we only use 'n.txt'
        # since the other files end with a number.txt
        if (file.endswith('n.txt')):
            lines = [line.rstrip('\n') for line in f]
            for line in lines:
                if (line.endswith('.fits') and (line not in allList)):
                    allList.append(line)
        else:
            lines = [line.rstrip('\n') for line in f]
            for line in lines:
                if (line.endswith('.fits') and (line not in tpList)):
                    tpList.append(line)
        f.close()
        examinedFiles += 1
        if ((examinedFiles % 25) == 0):
            print(str(examinedFiles) + ' files have been examined')

# False Positives are all FFs minus True Positives
fpList = [i for i in allList if i not in tpList]
print('The Number Of True positives Is: ', len(tpList))
print('The Number Of False positives Is: ', len(fpList))

# Now that we have the lists with all the files sorted we can write them on the .txt files:
for i in tpList:
    tp.write(i + '\r')

for j in fpList:
    fp.write(j + '\r')

# Close the files
tp.close()
fp.close()
