import os
dict = {}
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir,"song.txt")
file = open(file_path)
for line in file:
    line=line.rstrip()
    words=line.split()
    for word in words:
        if word in dict.keys():
            dict[word] += 1
        else:
            dict[word] = 1
file.close()
for word in dict.keys():
    print(word,":",dict[word])