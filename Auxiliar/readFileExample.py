








file = 'D:\MeteorDataBase\ServerData\labels\FTPdetectinfo_BE0001_20181209_161939_640835.txt'
f = open(file, 'r')
lines = [line.rstrip('\n') for line in f]
print(lines)
for element in lines:
    if element.endswith('.fits'):
        print(element)