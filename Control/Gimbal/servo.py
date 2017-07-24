import serial
import string

def send(pitch, roll, switch):
  if(switch == False):
    rot13 = str(pitch)+" "+str(roll)+" "
  else:
    rot13 = "-256 -256 "
  test = serial.Serial("/dev/ttyACM0", 115200, timeout=10)
  test.write(rot13)
  test.close()

try:
  while True:
    send(30, 49, True)
except KeyboardInterrupt:
    pass
