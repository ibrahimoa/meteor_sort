import os
from os.path import isdir, isfile, join




dir = 'D:\MeteorDataBase\ServerData\\nl0009\\files'
Filesfolder = 'processed'
for root, dirs, files in os.walk(join(dir, Filesfolder)):
    for folder in dirs:
        for file in os.listdir(join(dir, Filesfolder, folder)):
            if (file.endswith('.fits') or file.endswith('.txt')):
                pass
            else:
                os.remove(join(dir, Filesfolder, folder, file))
        print('Folder: ' + str(folder) + ' has been cleaned successfully')

