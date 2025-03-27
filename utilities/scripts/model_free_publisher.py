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
joints[0, 0] = torch.tensor([
    0.000147921,
    -0.78543,
    -6.659e-05,
    -2.35617,
    -3.26951e-05,
    1.57096,
    0.785415,
])

prev_time = -1

def callback_estimate(msg):
    global prev_time
    global joints

    # TODO validate first connection
    #      make sure we are not starting at an offset


    # TODO uses previous delta time
    # Ensure the first delta time is not incredebly large
    if prev_time == -1:
        prev_time = rospy.Time.now().to_sec()
        return

    # msg = rospy.wait_for_message('/model_free_controller/torques',
    #                              JointPoseStamped)
    tau = msg.pose
    torques = torch.Tensor(
        [tau.j1, tau.j2, tau.j3, tau.j4, tau.j5, tau.j6, tau.j7]
    ).unsqueeze(0)
    torques = torques.to(DEVICE)

    next_time = rospy.Time.now().to_sec()
    delta_time = next_time - prev_time

    print(f'delta time: {delta_time}')

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
    joint_publisher = rospy.Publisher('/model_free_controller/joint_estimates',
                                      JointEstimatesStamped, queue_size=10)
    
    torque_subscriber = rospy.Subscriber(
        name='/model_free_controller/torques',
        data_class=JointPoseStamped,
        callback=callback_estimate,
        queue_size=1
    )
    # rospy.Timer(rospy.Duration(0.03),
    #             callback_estimate)
    
    print('Loaded')
    rospy.spin()
