import tkinter
import serial

m=tkinter.Tk()

port = "COM7"
baud = 9600

ser = serial.Serial(port, baud, timeout=1)

#def m.destroy()
#    ser.close()

def OnePlus():
    ser.write(b'1,+' + bytes([10]))#bytes([13, 10]))

def OneMinus():
    ser.write(b'1,-' + bytes([10]))#bytes([13, 10]))

def TwoPlus():
    ser.write(b'2,+' + bytes([10]))#bytes([13, 10]))

def TwoMinus():
    ser.write(b'2,-' + bytes([10]))#bytes([13, 10]))

# open the serial port
if ser.isOpen():
    print(ser.name + ' is open...')

button1 = tkinter.Button(m, text='Exit', width=20, command=m.destroy)
button1.place(x = 400,y = 0)
button2 = tkinter.Button(m, text='1 +', width=20, command=OnePlus)
button2.place(x = 10,y = 0)
button3 = tkinter.Button(m, text='1 -', width=20, command=OneMinus)
button3.place(x = 150,y = 0)
button4 = tkinter.Button(m, text='2 +', width=20, command=TwoPlus)
button4.place(x = 10,y = 20)
button5 = tkinter.Button(m, text='2 -', width=20, command=TwoMinus)
button5.place(x = 150,y = 20)

m.mainloop()