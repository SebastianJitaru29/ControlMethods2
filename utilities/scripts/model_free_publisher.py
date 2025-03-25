#!/usr/bin/env python3

from networks.lagrangiannetwork import LagrangianNetwork
import torch
import rospy
from robot_msgs.msg import JointEstimatesStamped, JointPoseStamped

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH = '/home/student19/catkin_ws/src/utilities/scripts/networks/models/best_model.pt'

lnn = LagrangianNetwork(device=DEVICE)

lnn.load_state_dict(torch.load(PATH))
lnn.eval()

joints = torch.zeros((1, 2, 7), dtype=torch.float32, device=DEVICE)

prev_time = 0

def callback_estimate(msg):
    global prev_time
    global joints
    msg = rospy.wait_for_message("/model_free_controller/torques",
                                 JointPoseStamped)
    tau = msg.pose
    torques = torch.Tensor(
        [tau.j1, tau.j2, tau.j3, tau.j4, tau.j5, tau.j6, tau.j7]
    ).unsqueeze(0)
    torques = torques.to(DEVICE)

    next_time = rospy.Time.now().to_sec()
    delta_time = next_time - prev_time
    prev_time = next_time

    joints[0, 0] = joints[0, 0] / torch.pi
    joints[0, 1] = joints[0, 1] / (torch.pi * 2)

    with torch.no_grad():
        joints = lnn(joints, torques, delta_time)
    
    joints[0, 0] = joints[0, 0] * torch.pi
    joints[0, 1] = joints[0, 1] * torch.pi * 2
    
    joints_tmp = joints.cpu().squeeze(0)
    joints_array = torch.reshape(joints_tmp, (1, -1)).squeeze(0).numpy()
    joint_msg = JointEstimatesStamped()

    joints_array = [
        val for val in joints_array
    ]

    joint_msg.estimates.estimates = joints_array

    joints = joints.float()

    joint_publisher.publish(joint_msg)



if __name__ == '__main__':
    rospy.init_node("model_free_pub")
    prev_time = rospy.Time.now().to_sec()
    joint_publisher = rospy.Publisher('/model_free_controller/joint_estimates',
                                      JointEstimatesStamped, queue_size=10)
    
    rospy.Timer(rospy.Duration(0.03),
                callback_estimate)
    
    print('Loaded')
    rospy.spin()
