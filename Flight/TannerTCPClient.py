# By Tanner Winkelman
# start the server first, then this client can connect to it

import socket
from threading import Thread
"""
clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(('localhost', 8089))
clientsocket.send('Tanner says this message was sent from the client')

input( "Type '1234' and Press enter to go on: " )
"""

PORT_NUMBER = 8089


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', PORT_NUMBER))


def Reciever():
    buf = ""
    while buf != 'q' and buf != 'Q':
        buf = client_socket.recv(64)
        if( buf != 'q' and buf != 'Q' ):
            print buf

def Main():
  
  
    t1 = Thread(target=Reciever, args=())
    t1.start()
    
    data = ""
    while data != 'Q' and data != 'q':
        data = raw_input ( "SEND( TYPE q or Q to Quit):" )
        client_socket.send(data)
    client_socket.close()


if __name__ == '__main__':
    Main()
