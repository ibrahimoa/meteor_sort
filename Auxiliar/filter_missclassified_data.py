import os
from os.path import join
from shutil import copyfile

SRC_MET = "C:\work_dir\meteorData\extra_data\meteors"
SRC_NON_MET = "C:\work_dir\meteorData\extra_data\\non_meteors"
problematicFile = join('G:\GIEyA\TFG\meteor_classification\\results_2', 'problematicData_30.txt')

DST_MET = 'C:\work_dir\meteorData\extra_data_filtered_30\meteors'
DST_NON_MET = 'C:\work_dir\meteorData\extra_data_filtered_30\\non_meteors'

def FileToList(path):
    list = []
    with open(path, 'r') as f:
        for line in [line.rstrip('\n') for line in f]:
                list.append(line + '.jpg')
    return list

def moveWithFilter(SRC, DST, FILTER):
    contentList = os.listdir(SRC)
#     for i in range(len(contentList)):
    #         if os.path.getsize(os.path.join(SRC, contentList[i])) == 0:
    #             contentList.pop(i)
    for i in range(len(contentList)):
        if(i % 100 == 0):
            print(i)
        try:
            if contentList[i] not in FILTER:
                copyfile(os.path.join(SRC, contentList[i]), os.path.join(DST, contentList[i]))
            else:
                FILTER.pop(contentList[i])
        except:
            pass

# Problematic meteors: 2016
# Problematic non-meteors: 650
# Problematic all : 2666

FILTER = FileToList(problematicFile)
print(len(FILTER))
moveWithFilter(SRC_MET, DST_MET, FILTER)
moveWithFilter(SRC_NON_MET, DST_NON_MET, FILTER)
