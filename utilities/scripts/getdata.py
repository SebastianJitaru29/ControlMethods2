#!/usr/bin/env python

import rospy
import tf.transformations
import numpy as np

from geometry_msgs.msg import PoseStamped
from franka_msgs.msg import FrankaState
from sensor_msgs.msg import JointState

import os
import sys

DIR_PATH = '/home/student19/catkin_ws/src/data'
HEADER = 'j1,j2,j3,j4,j5,j6,j7,sec,nsec\n'

FILE_POS = None
FILE_VEL = None
FILE_EFF = None


def closehook():
    print('Shutdown Time!')
    FILE_POS.close()
    FILE_VEL.close()
    FILE_EFF.close()

def callback(msg):
    secs = str(msg.header.stamp.secs)
    nsecs = str(msg.header.stamp.nsecs)
    time_str = ',' + secs + ',' + nsecs + '\n'

    FILE_POS.write(','.join(map(str, msg.position)))
    FILE_POS.write(time_str)

    FILE_VEL.write(','.join(map(str, msg.velocity)))
    FILE_VEL.write(time_str)

    FILE_EFF.write(','.join(map(str, msg.effort)))
    FILE_EFF.write(time_str)

def listener():
    rospy.init_node('joint_listener')

    rospy.Subscriber("/franka_state_controller/joint_states", JointState, callback)

    rospy.on_shutdown(closehook)
    rospy.spin()

if __name__ == '__main__':
    DIR_PATH = '/home/student19/catkin_ws/src/data'
    trajectory_id = len(os.listdir(DIR_PATH))
    traj_path = DIR_PATH + '/traj_' + str(trajectory_id)
    traj_dir = os.mkdir(traj_path)

    FILE_POS = open(traj_path + '/positions', 'w')
    FILE_POS.write(HEADER)

    FILE_VEL = open(traj_path + '/velocities', 'w')
    FILE_VEL.write(HEADER)

    FILE_EFF = open(traj_path + '/efforts', 'w')
    FILE_EFF.write(HEADER)

    listener()
