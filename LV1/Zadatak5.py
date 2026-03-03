import os
hamWordCount = 0
hamLineCount = 0
spamWordCount = 0
spamLineCount = 0
spamExclamationCount = 0
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir,"SMSSpamCollection.txt")
file = open(file_path)
for line in file:
    line=line.rstrip()
    words=line.split()
    if words[0] == "ham":
        hamLineCount+=1
        for word in words:
            if word!=words[0]:
                hamWordCount+=1
    if words[0] == "spam":
        spamLineCount+=1
        for word in words:
            if word!=words[0]:
                spamWordCount+=1
        if line.endswith("!"):
            spamExclamationCount+=1
file.close()
print("Prosječan broj riječi ham poruke:", hamWordCount/hamLineCount)
print("Prosječan broj riječi spam poruke:", spamWordCount/spamLineCount)
print("Broj spam poruka s uskličnikom:", spamExclamationCount)