try:
    ocjena = float(input("Unesite broj između 0.0 i 1.0:"))
    if ocjena>1.0 or ocjena<0.0:
         raise IndexError
    if(ocjena>=0.9):
        print("A")
    elif(ocjena>=0.8):
        print("B")
    elif(ocjena>=0.7):
        print("C")
    elif(ocjena>=0.6):
        print("D")
    else:
        print("F")
except ValueError:
    print("Krivi tip podatka!")
    pass
except IndexError:
    print("Uneseni broj je van dosega!")
    pass