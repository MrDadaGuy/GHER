#!/usr/bin/env python


import sys, math, time, random
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import actionlib_msgs.msg
import geometry_msgs.msg
import trajectory_msgs.msg
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image, CompressedImage
import gym

sys.path.append("../rbx1_driver")
#import generate_samples

ARM_FAR_REACH = 0.44412
ARM_CLOSE_REACH = 0.21

moveit_commander.roscpp_initialize(sys.argv)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

group_arm = moveit_commander.MoveGroupCommander("arm")
group_gripper = moveit_commander.MoveGroupCommander("gripper")

display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=1)

class Sim:        #gym.core.Env
    """OpenAI Gym API for ROS# / Unity"""
    def __init__(self):

        rospy.init_node("pickle_rick")
        self.goal_pub = rospy.Publisher("move_group/goal", moveit_msgs.msg.MoveGroupActionGoal, queue_size=1)
        self.ros_time = rospy.Time()


    def step(self, new_ee_pos, gripper_pct_open):
#        assert new_ee_pos is type(list), "New EE position needs to be a list"
        assert len(new_ee_pos) == 3, "New EE position needs to have 3 elements: X, Y, Z"
        # move the arm, and then calculate reward
#        self._move_arm(self, new_ee_pos)

        print("gripper pct open = {}".format(gripper_pct_open))
        self._move_gripper(gripper_pct_open)




        obs = None
        reward = 0
        done = False
        info = {}

        return obs, reward, done, info


    def _move_arm(self, x, y, z=0.2):
        assert is_inside_circle(x, y, ARM_FAR_REACH), "Target is outside of RBX1 arm reach of {} m".format(ARM_FAR_REACH)
        assert not is_inside_circle(x, y, ARM_CLOSE_REACH), "Target is too close to RBX1 base, needs to be greater than {} m".format(ARM_CLOSE_REACH)

        group_arm.set_start_state_to_current_state()

        pose_target = geometry_msgs.msg.Pose()
        pose_target.orientation.w = -0.5
        pose_target.orientation.x = 0.5
        pose_target.orientation.y = 0.5
        pose_target.orientation.z = -0.5

        pose_target.position.x = x
        pose_target.position.y = y 
        pose_target.position.z = z
        group_arm.set_pose_target(pose_target)

        plan = group_arm.plan()

        print("============ Waiting while RVIZ displays plan1...")
        rospy.sleep(5)

        plan = group_arm.go(wait=True)

        group_arm.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        group_arm.clear_pose_targets()

    def _move_gripper(self, pct_open=1.0):

        assert pct_open >= 0.0 and pct_open <= 1.0

        joint_pos = (pct_open if pct_open < 1.0 else 0.999) * (math.pi / 2)

        joints = ['Joint_Grip_Idle', 'Joint_Tip_Idle', 'Joint_Grip_Idle_Arm', 'Joint_Grip_Servo', 'Joint_Tip_Servo', 'Joint_Grip_Servo_Arm']    # 'Joint_Grippper', 
        joint_positions = [joint_pos, joint_pos*-1, joint_pos, joint_pos*-1, joint_pos, joint_pos*-1]

        grip_goal = moveit_msgs.msg.MoveGroupActionGoal()

        header = Header()
        header.stamp = self.ros_time.now()
        goal_id = actionlib_msgs.msg.GoalID()
        goal_id.stamp = self.ros_time.now()
        goal_id.id = "ros_unity_sim-{}".format(goal_id.stamp)
        goal = moveit_msgs.msg.MoveGroupGoal()
        goal.request.workspace_parameters.header.stamp = self.ros_time.now()
        goal.request.workspace_parameters.header.frame_id = "base_footprint"
        goal.request.workspace_parameters.min_corner.x = -1.0
        goal.request.workspace_parameters.min_corner.y = -1.0
        goal.request.workspace_parameters.min_corner.z = -1.0
        goal.request.workspace_parameters.max_corner.x = 1.0
        goal.request.workspace_parameters.max_corner.y = 1.0
        goal.request.workspace_parameters.max_corner.z = 1.0
        goal.request.group_name = "gripper"
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 1.0
        goal.request.max_acceleration_scaling_factor = 1.0
        goal.request.start_state.is_diff = True

        constraints = moveit_msgs.msg.Constraints()

        for i in range(6):
            jc = moveit_msgs.msg.JointConstraint() 
            jc.joint_name = joints[i]
            jc.position = joint_positions[i]
            jc.tolerance_above = 0.0001
            jc.tolerance_below = 0.0001
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        goal.request.goal_constraints.append(constraints)

        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        goal.planning_options.planning_scene_diff.is_diff = True

        goal.planning_options.replan_delay = 2.0

        grip_goal.header = header
        grip_goal.goal_id = goal_id
        grip_goal.goal = goal

        while self.goal_pub.get_num_connections == 0:
            rospy.sleep(0.1)

        try:
            self.goal_pub.publish(grip_goal)
        except Exception as err:
            print(err)


    def render(self):
        # not sure what this means in our context, it'll always be rendered in unity.  whether we like it or not :(
        pass

    def reset(self):
        # Reset robot to original pose and regenerate new world, new random position of world objects in unity
        joint_goal = group_arm.get_current_joint_values()
        joint_goal[0] = 0.0
        joint_goal[1] = 0.0 # math.pi/3
        joint_goal[2] = 0.0 # math.pi/3
        joint_goal[3] = 0.0
        joint_goal[4] = 0.0 # math.pi/3
        joint_goal[5] = 0.0

        rearry_great_pran = group_arm.go(joint_goal, wait=True)
        group_arm.execute(rearry_great_pran, wait=True)
        group_arm.stop()

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

    ee_link = group_arm.get_end_effector_link()

    print("============ Reference frame: %s" % group_arm.get_planning_frame())
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
    group_arm.set_pose_target(pose_target)

    plan1 = group_arm.plan()

    print("============ Waiting while RVIZ displays plan1...")
    rospy.sleep(5)


    print("============ Visualizing plan1")
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()

    display_trajectory.trajectory_start = robot.get_current_state()
    display_trajectory.trajectory.append(plan1)
    display_trajectory_publisher.publish(display_trajectory)

    print("============ Waiting while plan1 is visualized (again)...")
#    rospy.sleep(5)

    # Uncomment below line when working with a real robot
    # group_arm.go(wait=True)

    group_variable_values = group_arm.get_current_joint_values()
    print("============ Joint values: ", group_variable_values)

    group_variable_values[0] = 1.0
    group_arm.set_joint_value_target(group_variable_values)

    plan2 = group_arm.plan()

    print("============ Waiting while RVIZ displays plan2...")
#    rospy.sleep(5)

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








#if __name__ == "__main__":
#
#    sim = Sim()
#    while True:
#        sim.move_gripper()
#        rospy.sleep(5)



#move_gripper()



#_move_arm(-0.312, -0.0312)

#time.sleep(5)

#_move_arm(0.25, 0.25, 0.2)

'''
- Joint_Grip_Servo
- Joint_Tip_Servo
- Joint_Grip_Servo_Arm
- Joint_Grip_Idle
- Joint_Tip_Idle
- Joint_Grip_Idle_Arm
position: [ -1.4791462193511427, 1.4791462193511427, -1.4791462193511427, 1.4791462193511427, -1.4791462193511427, 1.4791462193511427]
'''