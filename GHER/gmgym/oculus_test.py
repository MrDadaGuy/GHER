#!/usr/bin/env python

import rospy
from ros_unity_sim import Sim
import geometry_msgs.msg



def callback(data):
    rospy.loginfo("I heard %s",data)
    grip_open = data.pose.orientation.x    # need to convert to Euler befoer anything usefl
    grip_open = (grip_open + 1 ) / 2
    sim.step([0,0,0], grip_open)
    
def listener():
#    rospy.init_node('oculus_robot_control')
    rospy.Subscriber("/unity/controllers/left", geometry_msgs.msg.PoseStamped, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()



sim = Sim()

listener()

