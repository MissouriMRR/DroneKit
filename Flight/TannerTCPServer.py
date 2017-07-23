# start this first, then the client


import socket
from threading import Thread


serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind(('localhost', 8089))
serversocket.listen(5) # become a server socket, maximum 5 connections

connection, address = serversocket.accept()


def Reciever():
    buf = ""
    while buf != 'q' and buf != 'Q':
        buf = connection.recv(64)
        if( buf != 'q' and buf != 'Q' ):
            print buf


def Main():
  
  
    
    t1 = Thread(target=Reciever, args=())
    t1.start()
    
    data = ""

    while data != 'q' and data != 'Q':
        data = raw_input ( "SEND( TYPE q or Q to Quit):" )
        connection.send(data)

    connection.close()


if __name__ == '__main__':
    Main()
