#!/usr/bin/env python3


import zmq
import sys
import pickle

port = "5556"
if len(sys.argv) > 1:
    port =  sys.argv[1]
    int(port)


class Sim:

    def __init__(self):

        context = zmq.Context()
        print("Connecting to server...")
        self.socket = context.socket(zmq.REQ)
        self.socket.connect ("tcp://localhost:%s" % port)

    def send(self, request_message):

        print("SimProxy send...")

        self.socket.send(pickle.dumps(request_message))
        #  Get the reply.
        response_message = self.socket.recv()
        print("Received reply [", response_message, "]")


    def reset(self, request_message):
        print("SimProxy reset...")
