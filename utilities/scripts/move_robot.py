#!/usr/bin/env python

import rospy
import tf.transformations
import numpy as np

from geometry_msgs.msg import PoseStamped
from franka_msgs.msg import FrankaState

from copy import deepcopy

class Mover:

    def __init__(self):
        rospy.init_node("equilibrium_pose_node")
        self.pose_pub = rospy.Publisher("/cartesian_impedance_example_controller/equilibrium_pose", PoseStamped,
                                queue_size=10)
        """
        self.pose_pub = rospy.Publisher("/pd_grav_controller/joint_pose", PoseStamped,
                                        queue_size=10)
        """
        self.link_name = "" #rospy.get_param("~link_name")
        self.pose = PoseStamped()

        self.wait_for_initial_pose()
        self.start_pose = deepcopy(self.pose)
        print(self.start_pose)
        self.target_pose = deepcopy(self.pose)
        self.delta_idx = -1
    

    def run_sequence(self, deltas):
        
        self.wait_for_initial_pose()

        error = (self.pose.pose.position.x - self.target_pose.pose.position.x) + (self.pose.pose.position.y - self.target_pose.pose.position.y) + (self.pose.pose.position.z - self.target_pose.pose.position.z)

        if error < 0.0005:
            self.delta_idx += 1
            if self.delta_idx >= len(deltas):
                self.target_pose = self.start_pose
            else:
                self.target_pose = deepcopy(self.pose)
                self.target_pose.pose.position.x += deltas[self.delta_idx][0]
                self.target_pose.pose.position.y += deltas[self.delta_idx][1]
                self.target_pose.pose.position.z += deltas[self.delta_idx][2]  

        self.publish_pose(self.target_pose)

    def publish_pose(self, pose):
        pose.header.frame_id = self.link_name
        pose.header.stamp = rospy.Time(0)
        self.pose_pub.publish(pose)


    def wait_for_initial_pose(self):
        msg = rospy.wait_for_message("/franka_state_controller/franka_states",
                                    FrankaState)  # type: FrankaState

        initial_quaternion = \
            tf.transformations.quaternion_from_matrix(
                np.transpose(np.reshape(msg.O_T_EE,
                                        (4, 4))))
        initial_quaternion = initial_quaternion / \
            np.linalg.norm(initial_quaternion)
        self.pose.pose.orientation.x = initial_quaternion[0]
        self.pose.pose.orientation.y = initial_quaternion[1]
        self.pose.pose.orientation.z = initial_quaternion[2]
        self.pose.pose.orientation.w = initial_quaternion[3]
        self.pose.pose.position.x = msg.O_T_EE[12]
        self.pose.pose.position.y = msg.O_T_EE[13]
        self.pose.pose.position.z = msg.O_T_EE[14]



if __name__ == "__main__":
    # listener = tf.TransformListener()
    
    mover = Mover()
    

    seq = [
        [1, 1, 1],            # Starting position
    ]

    
    rospy.Timer(rospy.Duration(0.005),
                lambda msg: mover.run_sequence(seq))

    rospy.spin()


