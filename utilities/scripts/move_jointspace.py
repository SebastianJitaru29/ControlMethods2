#!/usr/bin/env python

import rospy
import tf.transformations
import numpy as np

from robot_msgs.msg import JointPoseStamped
from franka_msgs.msg import FrankaState

from copy import deepcopy

class Mover:

    def __init__(self, constraints):
        rospy.init_node("equilibrium_pose_node")
        self.pose_pub = rospy.Publisher("/pd_grav_controller/joint_pose", JointPoseStamped,
                                        queue_size=10)
        self.link_name = "" #rospy.get_param("~link_name")
        self.pose = JointPoseStamped()

        self.wait_for_initial_pose()
        self.start_pose = deepcopy(self.pose)
        self.target_pose = deepcopy(self.pose)
        self.delta_idx = -1

        self.n_joints = len(constraints)
        self.constraints = constraints
        self.ranges = constraints[:, 1] - constraints[:, 0]
        self.error_count = 0
    

    def run_random(self):

        self.wait_for_initial_pose()
        errors = self._get_error()

        print('max error: ', np.max(errors))
        print('-----')
        print(self.pose)
        print('-----')
        print(self.target_pose)
        print('-----')

        if (errors < 0.1).all() or self.error_count == 1:
            self.error_count = 0
            pose = self.create_random_q_d()
            print(pose)
            self.target_pose.pose.j1 = pose[0]
            self.target_pose.pose.j2 = pose[1]
            self.target_pose.pose.j3 = pose[2]
            self.target_pose.pose.j4 = pose[3]
            self.target_pose.pose.j5 = pose[4]
            self.target_pose.pose.j6 = pose[5]
            self.target_pose.pose.j7 = pose[6]

            self.publish_pose(self.target_pose)
        else:
            self.error_count += 1


    def run_sequence(self, deltas):
        
        self.wait_for_initial_pose()

        errors = self._get_error()

        print('max error: ', np.max(errors))
        
        if (errors < 0.3).all() or self.delta_idx == -1:
            print('After error')
            self.delta_idx += 1
            if self.delta_idx >= len(deltas):
                self.target_pose = self.start_pose
                print('Resetting')
            else:
                # TODO generate random joint space position based on limits
                self.target_pose = deepcopy(self.pose)
                self.target_pose.pose.j1 = deltas[self.delta_idx][0]
                self.target_pose.pose.j2 = deltas[self.delta_idx][1]
                self.target_pose.pose.j3 = deltas[self.delta_idx][2]
                self.target_pose.pose.j4 = deltas[self.delta_idx][3]
                self.target_pose.pose.j5 = deltas[self.delta_idx][4]
                self.target_pose.pose.j6 = deltas[self.delta_idx][5]
                self.target_pose.pose.j7 = deltas[self.delta_idx][6]
                print('Target pose',self.target_pose)

        self.publish_pose(self.target_pose)

    def publish_pose(self, pose):
        pose.header.frame_id = self.link_name
        pose.header.stamp = rospy.Time(0)
        self.pose_pub.publish(pose)

    def _get_error(self):
        errors = np.zeros(7)
        errors[0] = np.abs(self.pose.pose.j1 - self.target_pose.pose.j1)
        errors[1] = np.abs(self.pose.pose.j2 - self.target_pose.pose.j2)
        errors[2] = np.abs(self.pose.pose.j3 - self.target_pose.pose.j3)
        errors[3] = np.abs(self.pose.pose.j4 - self.target_pose.pose.j4)
        errors[4] = np.abs(self.pose.pose.j5 - self.target_pose.pose.j5)
        errors[5] = np.abs(self.pose.pose.j6 - self.target_pose.pose.j6)
        errors[6] = np.abs(self.pose.pose.j7 - self.target_pose.pose.j7)

        return errors

    def create_random_q_d(self):
        rnd = np.random.rand(self.n_joints)
        q_d = rnd * self.ranges + self.constraints[:, 0]
        return q_d

    def wait_for_initial_pose(self):
        msg = rospy.wait_for_message("/franka_state_controller/franka_states",
                                    FrankaState)  # type: FrankaState

        self.pose.pose.j1 = msg.q[0]
        self.pose.pose.j2 = msg.q[1]
        self.pose.pose.j3 = msg.q[2]
        self.pose.pose.j4 = msg.q[3]
        self.pose.pose.j5 = msg.q[4]
        self.pose.pose.j6 = msg.q[5]
        self.pose.pose.j7 = msg.q[6]

if __name__ == "__main__":
    # listener = tf.TransformListener()
    

    seq = [
        # Upper loop of the "8"
        [0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0],   # Start position (upper part of the 8)
        [0.0, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0]
    ]
    limits = np.array([
        [-2.8973, 2.8973],
        [-1.7627, 1.7627],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ])

    mover = Mover(limits)
    
    rospy.Timer(rospy.Duration(2.5),
                lambda msg: mover.run_random())

    rospy.spin()


