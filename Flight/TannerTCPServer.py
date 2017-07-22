# start this first, then the client


import socket

serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind(('localhost', 8089))
serversocket.listen(5) # become a server socket, maximum 5 connections

connection, address = serversocket.accept()

data = ""

while data != 'q' and data != 'Q':
    buf = connection.recv(64)
    if len(buf) > 0:
        print buf
        data = raw_input ( "SEND( TYPE q or Q to Quit):" )
        if(data != 'q' and data != 'Q'):
            connection.send(data)
