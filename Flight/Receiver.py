import socket
from threading import Thread
import pickle
from PIL import Image
from matplotlib import pyplot as plt


# needs to be the same on the client
PORT_NUMBER = 8089

UP_BOARD_IP = 'localhost' #'192.168.12.1' actually I don't know what address to put here



global color_image
global depth_image

import socket
import numpy as np
from cStringIO import StringIO

class numpysocket():
    def __init__(self):
        pass

    @staticmethod
    def startServer():
        port=PORT_NUMBER
        server_socket=socket.socket() 
        server_socket.bind(('',port))
        server_socket.listen(1)
        print 'waiting for a connection...'
        client_connection,client_address=server_socket.accept()
        print 'connected to ',client_address[0]
        ultimate_buffer=''
        while True:
            receiving_buffer = client_connection.recv(1024)
            if not receiving_buffer: break
            ultimate_buffer+= receiving_buffer
            print '-',
        final_image=np.load(StringIO(ultimate_buffer))['frame']
        client_connection.close()
        server_socket.close()
        print '\nframe received'
        return final_image

   


def Receiver():
    buf = ""
    while True:
        color = numpysocket.startServer()
        #size = int( connection.recv(6) )
        #print size
        #buf = connection.recv(size)
        #print buf
        #color = pickle.loads(buf)
        
        plt.imshow(color, interpolation='nearest')
        plt.show()
        print color

def get_last_color_frame():
    return color_image

def get_last_depth_frame():
    return depth_image



def Main():
    print( "beginning of main" )
    t1 = Thread(target=Receiver, args=())
    print( "middle of main" )
    t1.start()
    print( "end of main" )

    



if __name__ == '__main__':
  Main()
