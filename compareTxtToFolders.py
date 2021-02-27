import os
from os.path import join

def detectionsFileToList(path):
    list = []
    f = open(path, 'r')
    for line in [line.rstrip('\n') for line in f]:
        if (line.endswith('.fits') and (line not in list)):
            list.append(line)
    f.close()
    return list
def checkListExclusion(list1, list2):
    return list(set(list1).intersection(list2))

folderDir = 'D:\MeteorDataBase\ServerData'
tp = 'truePositivesMeteors.txt'
fp = 'falsePositivesMeteors.txt'
tp_add = 'truePositivesMeteorsAdditional.txt'
fp_add = 'falsePositivesMeteorsAdditional.txt'

tp_all = 'new_meteors.txt'
fp_all = 'new_non_meteors.txt'

tpL = detectionsFileToList(join(folderDir, tp))
fpL = detectionsFileToList(join(folderDir, fp))
tp_addL = detectionsFileToList(join(folderDir, tp_add))
fp_addL = detectionsFileToList(join(folderDir, fp_add))

# Check that all lists are congruent:

fails_TP_FPAdd = checkListExclusion(tpL, fp_addL)
print(fails_TP_FPAdd)
fails_TPAdd_FP = checkListExclusion(tp_addL, fpL)
print(fails_TPAdd_FP)

fails_TP_FP = checkListExclusion(tpL, fpL)
print(fails_TP_FP)
fails_TPAdd_FPAdd = checkListExclusion(tp_addL, fp_addL)
print(fails_TPAdd_FPAdd)

common_tp = checkListExclusion(tpL, tp_addL)
print(len(common_tp))
common_fp = checkListExclusion(fpL, fp_addL)
print(len(common_fp))

# Create two new lists with all meteors and all non-meteors:
meteorsList = []
nonMeteorsList = []
for element in tp_addL:
    if(element not in tpL):
        meteorsList.append(element)
for element in fp_addL:
    if(element not in fpL):
        nonMeteorsList.append(element)

meteors = open(join(folderDir, tp_all), 'w+')
non_meteors = open(join(folderDir, fp_all), 'w+')

for i in meteorsList:
    meteors.write(i + '\r')
for i in nonMeteorsList:
    non_meteors.write(i + '\r')
print('-----------')
print(len(tpL))
print(len(fpL))
print(len(meteorsList))
print(len(nonMeteorsList))


