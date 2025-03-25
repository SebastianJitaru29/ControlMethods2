// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/PoseStamped.h>
#include <robot_msgs/JointPoseStamped.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <eigen3/Eigen/Dense>

#include <franka_example_controllers/compliance_paramConfig.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>

namespace model_free_controller {

class ModelFreeController : public controller_interface::MultiInterfaceController<
                                                franka_hw::FrankaModelInterface,
                                                hardware_interface::EffortJointInterface,
                                                franka_hw::FrankaStateInterface>
{
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:

  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  ros::Publisher torque_pub_;

  Eigen::Vector3d position_d_;
  Eigen::Quaterniond orientation_d_;
  std::mutex position_and_orientation_d_target_mutex_;
  Eigen::Vector3d position_d_target_;
  Eigen::Quaterniond orientation_d_target_;

  Eigen::Matrix<double, 7, 1> q_desired;

  Eigen::DiagonalMatrix<double, 7> k_p; // = Eigen::DiagonalMatrix<double, 7>(20.0, 30.0, 20.0, 10.0, 15.0, 25.0, 10.0);
  Eigen::DiagonalMatrix<double, 7> k_d; // = Eigen::DiagonalMatrix<double, 7>(5.0, 5.0, 5.0, 3.0, 3.0, 4.0, 3.0);

  // Equilibrium pose subscriber
  ros::Subscriber sub_joint_pose_;
  void jointPoseCallback(const robot_msgs::JointPoseStampedConstPtr& msg);
};

}  // namespace model_free_controller
