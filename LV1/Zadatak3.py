import math
list = []
while True:
    unos = input("Unesite broj ili Done ako ste gotovi:")
    if unos=="Done":
        break
    try:
        vrijednost = float(unos)
        list.append(vrijednost)
    except:
        print("Dani ulaz nije broj ni ne naznačava kraj!")
print("Količina unešenih brojeva:",len(list))
print("Prosjek brojeva liste:",sum(list)/len(list))
print("Najveći broj u listi:",max(list))
print("Najmanji broj u listi:",min(list))
print("Sortirana lista:",sorted(list))