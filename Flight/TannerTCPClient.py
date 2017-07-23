# By Tanner Winkelman
# start the server first, then this client can connect to it

import socket
from threading import Thread
import numpy as np
import pickle
import StringIO
import json

NUMPY_SIGNAL = "SERIALIZED_NUMPY_ARRAY_FOLLOWS"

a = np.arange(15).reshape(3, 5)

PORT_NUMBER = 8089


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', PORT_NUMBER))


def Receiver():
    buf = ""
    while buf != 'q' and buf != 'Q':
        buf = client_socket.recv(32768)
        if( buf != 'q' and buf != 'Q' ):
            print buf

def Main():
  
  
  
    t1 = Thread(target=Receiver, args=())
    t1.start()
    
    data = ""
    while data != 'Q' and data != 'q':
        data = raw_input ( "SEND( TYPE q or Q to Quit, TYPE n TO SEND NUMPY ARRAY):" )
        if data == 'n':

            serialized = pickle.dumps(a, protocol=0) # protocol 0 is printable ASCII
            print NUMPY_SIGNAL + serialized
            client_socket.sendall(NUMPY_SIGNAL + serialized)
        else:
            client_socket.send(data)
    client_socket.close()


if __name__ == '__main__':
    Main()
