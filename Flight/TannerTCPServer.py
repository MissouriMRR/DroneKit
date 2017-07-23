# start this first, then the client


import socket
from threading import Thread
import numpy as np
import pickle
import StringIO
import json


NUMPY_SIGNAL = "SERIALIZED_NUMPY_ARRAY_FOLLOWS"


serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind(('localhost', 8089))
serversocket.listen(5) # become a server socket, maximum 5 connections

connection, address = serversocket.accept()

NUMPY_ARRAY = np.arange(15).reshape(3,5)



def Receiver():
    buf = ""
    while buf != 'q' and buf != 'Q':
        buf = connection.recv(32768)
        
        if( buf != 'q' and buf != 'Q' ):
            if (NUMPY_SIGNAL in buf):
                buf = buf[len(NUMPY_SIGNAL):]
                a = pickle.loads(buf)
                print a
            else:
                print buf

            print("type:" + str(type( buf )))
            



def Main():
  
  
    
    t1 = Thread(target=Receiver, args=())
    t1.start()
    
    data = ""

    while data != 'q' and data != 'Q':
        data = raw_input ( "SEND( TYPE q or Q to Quit):" )
        connection.send(data)

    connection.close()


if __name__ == '__main__':
    Main()
