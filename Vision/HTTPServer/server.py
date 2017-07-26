from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pickle

import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.websocket
import base64

import threading

from time import sleep

from RealSense import Streamer

SERVER_CLOSING = 'Server has shut down'
LOG_PREFIX = 'SERVER:'
PORT_NUMBER = 8888

class SentinelServer(tornado.websocket.WebSocketHandler):
    stream = None

    def initialize(self):
        self.realsense_stream = SentinelServer.stream
      
    def send_frame(self):
        frame = self.realsense_stream.next()
        encoded_frame = pickle.dumps(frame, 2)
        self.write_message(encoded_frame, binary = True)
    
    def on_pong(self, pong):
        self.log('Pong:', pong)
    
    def open(self):
        self.stream.set_nodelay(True)
        self.ping(b'ping')
        self.send_frame()

    def log(self, *args):
        print(LOG_PREFIX, *args)

    def on_message(self, message):
        self.log(pickle.loads(message))
        self.send_frame()

    def on_close(self):
        self.log('Lost connection to client.')
  
if __name__ == '__main__':
    with Streamer() as stream:
        try:
            SentinelServer.stream = stream
            application = tornado.web.Application([(r'/ws', SentinelServer),])
            http_server = tornado.httpserver.HTTPServer(application)
            http_server.listen(PORT_NUMBER)
            tornado.ioloop.IOLoop.instance().start()
        except KeyboardInterrupt as interrupt:
            pass
