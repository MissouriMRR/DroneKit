import socket
import cv2
import numpy

def send_image(ip, port, image):
    """ Sends an image over socket using TCP.

    Args:
        ip - The ip address to connect to.
        port - The port to connect to.
        image - The image being sent.
    """
    
    sock = socket.socket()
    sock.connect((ip, port))

    encode_image= [int(image.IMWRITE_JPEG_QUALITY), 90]
    result, image_encode = image.imencode('.jpg', frame, encode_image)
    data = numpy.array(image_encode)
    string_data = str(data)

    sock.send( str(len(stringData)).ljust(16))
    sock.send( stringData )
    sock.close()
