#!/usr/bin/env python


import sys, math, time
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image, CompressedImage
import gym

sys.path.append("../rbx1_driver")
import generate_samples



ARM_FAR_REACH = 0.44412
ARM_CLOSE_REACH = 0.21

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface_tutorial', anonymous=True)

robot = moveit_commander.RobotCommander()

scene = moveit_commander.PlanningSceneInterface()

#group = moveit_commander.MoveGroupCommander("arm")
group = moveit_commander.MoveGroupCommander("panda_arm")

display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                    moveit_msgs.msg.DisplayTrajectory, queue_size=10)

class Sim(gym.core.Env):
    """OpenAI Gym API for ROS# / Unity"""
    def __init__(self):
        pass


    def step(self, new_ee_pos):
        assert new_ee_pos is type(tuple), "New EE position needs to be a tuple"
        assert len(new_ee_pos) == 3, "New EE position needs to have 3 elements: X, Y, Z"
        # move the arm, and then calculate reward
        self._move_arm(*new_ee_pos)

        data, labels = generate_samples.generate()

        # step returns:  observation(obj), reward(float), done(bool), info(dict)
        # obs = XYZ of EE, XYZ of target
        obs = None
        reward = 0
        done = False
        info = {}

        return obs, reward, done, info


    def render(self):
        # not sure what this means in our context, it'll always be rendered in unity.  whether we like it or not :(
        pass

    def reset(self):
        # Reset robot to original pose and regenerate new world, new random position of world objects in unity
        joint_goal = group.get_current_joint_values()
        joint_goal[0] = 0.0
        joint_goal[1] = 0.0 # math.pi/3
        joint_goal[2] = 0.0 # math.pi/3
        joint_goal[3] = 0.0
        joint_goal[4] = 0.0 # math.pi/3
        joint_goal[5] = 0.0

        rearry_great_pran = group.go(joint_goal, wait=True)
        group.execute(rearry_great_pran, wait=True)
        group.stop()

        obs = None
        reward = 0
        done = False
        info = {}

        return obs, reward, done, info

    def close(self):
        pass

    class action_space():
        def sample(self):
            return [0.3, 0.3, 0.3]




def print_metadata():

    ee_link = group.get_end_effector_link()

    print("============ Reference frame: %s" % group.get_planning_frame())
    print("============ EE link: %s" % ee_link)

    print("============ Robot Groups:")
    print(robot.get_group_names())

    print("============ Printing robot state")
    print(robot.get_current_state())
    print("============")




def move_ee():

    print("============ Generating plan 1")
    pose_target = geometry_msgs.msg.Pose()
    pose_target.orientation.w = 1.0
    pose_target.position.x = 0.7
    pose_target.position.y = -0.5
    pose_target.position.z = 1.1
    group.set_pose_target(pose_target)

    plan1 = group.plan()

    print("============ Waiting while RVIZ displays plan1...")
    rospy.sleep(5)


    print("============ Visualizing plan1")
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()

    display_trajectory.trajectory_start = robot.get_current_state()
    display_trajectory.trajectory.append(plan1)
    display_trajectory_publisher.publish(display_trajectory)

    print("============ Waiting while plan1 is visualized (again)...")
    rospy.sleep(5)

    # Uncomment below line when working with a real robot
    # group.go(wait=True)

    group_variable_values = group.get_current_joint_values()
    print("============ Joint values: ", group_variable_values)


    group_variable_values[0] = 1.0
    group.set_joint_value_target(group_variable_values)

    plan2 = group.plan()

    print("============ Waiting while RVIZ displays plan2...")
    rospy.sleep(5)

def add_box(x, y, z):

    box_pose = geometry_msgs.msg.PoseStamped()
    box_pose.header.frame_id = "Link_0"
    box_pose.pose.orientation.w = 1.0
    box_pose.pose.position.x = x # slightly above the end effector
    box_pose.pose.position.y = y # slightly above the end effector
    box_pose.pose.position.z = z # slightly above the end effector
    box_name = "box"
    scene.add_box(box_name, box_pose, size=(0.1, 0.1, 0.1))
    #scene.remove_world_object(box_name)

def is_inside_circle(x, y, rad, circle_x = 0, circle_y = 0):
    if ((x - circle_x) * (x - circle_x) + (y - circle_y) * (y - circle_y) <= rad * rad):
        return True
    return False


def _move_arm(x, y, z=0.2):
#    assert is_inside_circle(x, y, ARM_FAR_REACH), "Target is outside of RBX1 arm reach of {} m".format(ARM_FAR_REACH)
#    assert not is_inside_circle(x, y, ARM_CLOSE_REACH), "Target is too close to RBX1 base, needs to be greater than {} m".format(ARM_CLOSE_REACH)

    group.set_start_state_to_current_state()

    pose_target = geometry_msgs.msg.Pose()
    pose_target.orientation.w = -0.5
    pose_target.orientation.x = 0.5
    pose_target.orientation.y = 0.5
    pose_target.orientation.z = -0.5

    pose_target.position.x = x
    pose_target.position.y = y 
    pose_target.position.z = z
    group.set_pose_target(pose_target)

    plan1 = group.plan()

    print "============ Waiting while RVIZ displays plan1..."
    rospy.sleep(5)

    plan = group.go(wait=True)

    group.stop()
    # It is always good to clear your targets after planning with poses.
    # Note: there is no equivalent function for clear_joint_value_targets()
    group.clear_pose_targets()


_move_arm(-0.312, -0.0312)

time.sleep(5)

_move_arm(0.25, 0.25, 0.2)


