# By Tanner Winkelman
# start the server first, then this client can connect to it

import socket
"""
clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(('localhost', 8089))
clientsocket.send('Tanner says this message was sent from the client')

input( "Type '1234' and Press enter to go on: " )
"""

PORT_NUMBER = 8089

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', PORT_NUMBER))
data = ""
while data <> 'Q' and data <> 'q':
    data = raw_input ( "SEND( TYPE q or Q to Quit):" )
    if (data <> 'Q' and data <> 'q'):
        client_socket.send(data)
        buf = client_socket.recv(64)
        print buf
