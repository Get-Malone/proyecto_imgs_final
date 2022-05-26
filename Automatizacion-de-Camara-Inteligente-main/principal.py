import os
from tkinter import *

def validate():
    value = option.get()
    if value == "cam":
        os.system('python Camara.py')
    elif value == "selecc":
        os.system('python Tracking.py')
    elif value == "cubre":
        os.system('python Facemask.py')
    else:
        print("An option must be selected")

root = Tk()
root.geometry("400x400")
root.eval('tk::PlaceWindow . center')

option = StringVar()
R1 = Radiobutton(root, text="CAMARA", value="cam", var=option)
R2 = Radiobutton(root, text="SELECCIONAR ROSTRO", value="selecc", var=option)
R3 = Radiobutton(root, text="CUBREBOCAS", value="cubre", var=option)
button = Button(root, text="OK", command=validate)

R1.pack()
R2.pack()
R3.pack()
button.pack()

root.mainloop()