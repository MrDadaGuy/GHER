#!/usr/bin/env python

import zmq
import time
import sys
import json
import pickle
sys.path.insert(0, '/home/modsim/GHER/')

import GHER.gmgym.ros_unity_sim
from sim_msgs import Observation

import sys, signal
def signal_handler(signal, frame):
    print("\nSimProxyListener received interrupt signal, exiting.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class SimProxyListener:

    def __init__(self):
        self.sim = GHER.gmgym.ros_unity_sim.Sim()
        port = "5556"
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://*:%s" % port)

        while True:
            #  Wait for next request from client
            message = self.socket.recv()
            print("Received request: ", message)
            message = pickle.loads(message)

            if message == "reset":
                obs, reward, done, info = self.sim.reset()
                print("Object Poses = " + str(obs))
#                self.target_position = obs
                self.socket.send(pickle.dumps(obs))
#                self.socket.send_string("Reset OK")     # REQ:REP needs a response!! client is waiting

# TODO:  Randomize position of object -- already have this in generate_samples

            time.sleep(0.001)


    def step(self, obs):
        pass        


if __name__ == "__main__":
    sim = SimProxyListener()


#        time.sleep (1)  
#        socket.send_string("Box x is %s" % obs.Box.x)