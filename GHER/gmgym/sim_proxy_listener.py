#!/usr/bin/env python3

import zmq
import time
import sys
import json
import pickle
import GHER.gmgym.ros_unity_sim
from sim_msgs import Observation



class SimProxyListener:

    port = "5556"
    if len(sys.argv) > 1:
        port =  sys.argv[1]
        int(port)


    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)
    #socket.bind("tcp://*:%s" % port)

    while True:
        #  Wait for next request from client
        message = socket.recv()
        print("Received request: ", message)
#        obs = Observation().fromJSON(message)
        obs = pickle.loads(message)

        time.sleep (1)  
        socket.send_string("Box x is %s" % obs.Box.x)