// #include "model_free_controller.h"

/*
Data needed:
 - joint positions
 - joint velocities
 - joint torques
 - delta time
*/

// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
// #include <pd_grav_controller/pd_grav_controller.h>

#include "../include/model_free_controller/model_free_controller.h"

#include <cmath>
#include <memory>
#include <iostream>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka_example_controllers/pseudo_inversion.h>

namespace pd_grav_controller {

bool PDGravController::init(hardware_interface::RobotHW* robot_hw,
                                               ros::NodeHandle& node_handle) {

  std::cout << "Initializing robot" << std::endl;
  std::vector<double> cartesian_stiffness_vector;
  std::vector<double> cartesian_damping_vector;

  sub_joint_pose_ = node_handle.subscribe(
      "/pd_grav_controller/joint_pose", 20, &PDGravController::jointPoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("PDGravController: Could not read parameter arm_id");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "PDGravController: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "PDGravController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle("panda_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "PDGravController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "PDGravController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle("panda_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "PDGravController: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "PDGravController: Error getting effort joint interface from hardware");
    return false;
  }

  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "PDGravController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  dynamic_reconfigure_compliance_param_node_ =
      ros::NodeHandle(node_handle.getNamespace() + "/dynamic_reconfigure_compliance_param_node");

  dynamic_server_compliance_param_ = std::make_unique<
      dynamic_reconfigure::Server<franka_example_controllers::compliance_paramConfig>>(

      dynamic_reconfigure_compliance_param_node_);
  dynamic_server_compliance_param_->setCallback(
      boost::bind(&PDGravController::complianceParamCallback, this, _1, _2));

  position_d_.setZero();
  orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  position_d_target_.setZero();
  orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;

  k_d.diagonal() << 30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0;

  k_p.diagonal() << 200.0, 200.0, 200.0, 200.0, 750.0, 50.0, 20.0;

  return true;
}

void PDGravController::starting(const ros::Time& /*time*/) {
  // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
  // to initial configuration
  franka::RobotState initial_state = state_handle_->getRobotState();
  // get jacobian
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  // convert to eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));

  // set equilibrium point to current state
  position_d_ = initial_transform.translation();
  orientation_d_ = Eigen::Quaterniond(initial_transform.rotation());
  position_d_target_ = initial_transform.translation();
  orientation_d_target_ = Eigen::Quaterniond(initial_transform.rotation());

  // set nullspace equilibrium configuration to initial q

  //q_desired << 0.0, 0.5, 0, -1.5, 0, 2.5, 0;
}

void PDGravController::update(const ros::Time& /*time*/,
                                                const ros::Duration& /*period*/) {
  // get state variables
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);

  // convert to Eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());

  std::array<double, 7> gravity_array = model_handle_->getGravity();
  Eigen::Map<Eigen::Matrix<double, 7, 1>> gravity(gravity_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_measured(robot_state.tau_J.data());

  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(  // NOLINT (readability-identifier-naming)
      robot_state.tau_J_d.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.rotation());
  

  // compute error to desired joint positions
  Eigen::Matrix<double, 7, 1> q_error = q_desired - q;
  Eigen::Matrix<double, 7, 1> q_error_kped = k_p * q_error;
  Eigen::Matrix<double, 7, 1> q_dot_kded = k_d * dq;
  Eigen::Matrix<double, 7, 1> u_before_grav = q_error_kped - q_dot_kded ;
  
  // aproximated mass 5 5 2 
  // length 0.3160 0.3840 0.0880 0.1070
 
  // TODO subtract gravity from final torques to get true effort?

  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(-gravity[i]);
  }
  // TODO add gravity compensation
}

void PDGravController::jointPoseCallback(const robot_msgs::JointPoseStampedConstPtr& msg)
{
  std::lock_guard<std::mutex> position_d_target_mutex_lock(
    position_and_orientation_d_target_mutex_);

  // q_desired << msg->pose.j1, msg->pose.j2, msg->pose.j3, msg->pose.j4, msg->pose.j5, msg->pose.j6, msg->pose.j7;
  // std::cout << "Desired angles: " << q_desired << std::endl;
}

}  // namespace pd_grav_controller

PLUGINLIB_EXPORT_CLASS(pd_grav_controller::PDGravController,
                       controller_interface::ControllerBase)
