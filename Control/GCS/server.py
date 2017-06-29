import socket
import cv2
import numpy

def recvall(sock, count):
""" I've got no idea what this one does. """
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def receive_image(ip, port):
    """ Receives an image being sent over port using TCP.

    Args:
        ip - A symbolic input for the port.
        port - The port being sent from.

    Returns:
        A rebuilt image.
    """
    
    socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(( ip, port ))
    s.listen(True)
    conn, addr = s.accept()
    length = recvall(conn, 16)
    string_data = recvall(conn, int(length))
    data = numpy.fromstring(string_data, dtype = 'uint8')
    s.close()

    image = cv2.imdecode(data, 1)
    return image
