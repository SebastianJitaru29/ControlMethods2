#!/usr/bin/env python

import rospy
import sys

from robot_msgs.msg import JointPoseStamped, JointPose
from franka_msgs.msg import FrankaState

from copy import deepcopy
import numpy as np


pub_pose = None

FILE = '/home/student19/catkin_ws/src/utilities/increased_kp/joint'

BASEPOSE = JointPoseStamped()

BASEPOSE.pose.j1 = 0
BASEPOSE.pose.j2 = -0.785
BASEPOSE.pose.j3 = 0
BASEPOSE.pose.j4 = -2.356
BASEPOSE.pose.j5 = 0
BASEPOSE.pose.j6 = 1.571
BASEPOSE.pose.j7 = 0.785

target_pose = deepcopy(BASEPOSE)

current_pose = deepcopy(BASEPOSE)

moving_joint = 0

info = np.zeros((2, 7))

record = False

arrived = False

target_angle = 0

def publish_joint_pose(j_idx, angle):
    global target_angle
    global arrived
    global target_pose

    target_angle = angle
    arrived = False

    if j_idx == 0:
        target_pose.pose.j1 += angle
    elif j_idx == 1:
        target_pose.pose.j2 += angle
    elif j_idx == 2:
        target_pose.pose.j3 += angle
    elif j_idx == 3:
        target_pose.pose.j4 += angle
    elif j_idx == 4:
        target_pose.pose.j5 += angle
    elif j_idx == 5:
        target_pose.pose.j6 += angle
    elif j_idx == 6:
        target_pose.pose.j7 += angle
        
    pub_pose.publish(target_pose)
    

def publish_pose(pose):
    global arrived
    arrived = False
    pub_pose.publish(pose)


def callback_state(msg):
    global current_pose
    global moving_joint
    global arrived
    global target_angle

    error = 0

    if moving_joint == 0:
        error = np.abs(target_pose.pose.j1 - msg.q[0])
    elif moving_joint == 1:
        error = np.abs(target_pose.pose.j2 - msg.q[1])
    elif moving_joint == 2:
        error = np.abs(target_pose.pose.j3 - msg.q[2])
    elif moving_joint == 3:
        error = np.abs(target_pose.pose.j4 - msg.q[3])
    elif moving_joint == 4:
        error = np.abs(target_pose.pose.j5 - msg.q[4])
    elif moving_joint == 5:
        error = np.abs(target_pose.pose.j6 - msg.q[5])
    elif moving_joint == 6:
        error = np.abs(target_pose.pose.j7 - msg.q[6])
    
    if error <= 0.005:
        arrived = True

    """
    Timestamp
    Q
    Qdot
    Error
    Target
    """
    joint_q = msg.q[moving_joint]
    joint_dq = msg.dq[moving_joint]

    if record:
        time = rospy.Time.now().to_sec()
        line = str(time) + ',' \
               + str(joint_q) + ',' \
               + str(joint_dq) + ',' \
               + str(error) + ',' \
               + str(target_angle) + '\n'
        
        with open(FILE, 'a') as csv_file:
            csv_file.write(line)

    # TODO log

if __name__ == '__main__':
    rospy.init_node('joint_mover_node')

    assert len(sys.argv) == 3

    try:
        joint_idx = int(sys.argv[1])
    except ValueError:
        print('provide integer for joint index')
        quit(1)

    try:
        angle = float(sys.argv[2])
    except ValueError:
        print('provide float for joint angle')
        quit(1)

    FILE += '_' + str(joint_idx) + '.csv'

    pub_pose = rospy.Publisher(
        name='/pd_grav_controller/joint_pose',
        data_class=JointPoseStamped,
        queue_size=10,
    )

    sub_state = rospy.Subscriber(
        name='/franka_state_controller/franka_states',
        data_class=FrankaState,
        callback=callback_state,
        queue_size=10,
    )
    rospy.sleep(2)

    publish_pose(target_pose)
    while not arrived:
        rospy.sleep(1)

    record = True

    with open(FILE, 'w') as csv_file:
        csv_file.write('time,q,qdot,error,target_angle\n')

    publish_joint_pose(joint_idx, angle)

    while not arrived:
        rospy.sleep(1)

    rospy.sleep(1)
    record = False
    target_pose = deepcopy(BASEPOSE)
    publish_pose(target_pose)
    while not arrived:
        rospy.sleep(1)
    print('!-- ALL DONE --!')
    rospy.spin()

