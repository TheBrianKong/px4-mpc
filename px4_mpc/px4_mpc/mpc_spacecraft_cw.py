#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2024 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

__author__ = "Pedro Roque, Jaeyoung Lim"
__contact__ = "padr@kth.se, jalim@ethz.ch"

import rclpy
import time
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from visualization_msgs.msg import Marker
from trajectory_msgs.msg import JointTrajectory

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleAngularVelocity
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleRatesSetpoint
from px4_msgs.msg import ActuatorMotors
from px4_msgs.msg import VehicleTorqueSetpoint
from px4_msgs.msg import VehicleThrustSetpoint

from mpc_msgs.srv import SetPose

from px4_mpc.utils.rotations import q_to_rot_mat_np

DATA_VALIDITY_STREAM = 0.5 # seconds, threshold for (pos,att,vel) messages
DATA_VALIDITY_STATUS = 2.0 # seconds, threshold for status message

class SpacecraftMPC(Node):

    def __init__(self):
        super().__init__('spacecraft_mpc')

        # Get mode; rate, wrench, direct_allocation
        self.mode = self.declare_parameter('mode', 'wrench').value
        self.sitl = self.declare_parameter('sitl', False).value
        self.use_ned = self.declare_parameter('px4_uses_ned', True).value
        self.orbit_period = self.declare_parameter('orbit_period', 90.0).value
        self.skip_build = self.declare_parameter('skip_build', False).value
        self.get_logger().info(f'MPC mode: {self.mode}, SITL: {self.sitl}, PX4 uses NED: {self.use_ned}, Orbit period: {self.orbit_period} min, Skip build: {self.skip_build}')

        # Get setpoint from rviz (true/false)
        self.setpoint_from_rviz = self.declare_parameter('setpoint_from_rviz', False).value

        # Camera mode (true/false)
        self.camera = self.declare_parameter('camera', False).value

        # QoS profile
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Setup publishers and subscribers
        self.set_publishers_subscribers(qos_profile)

        timer_period = 0.1  # seconds
        timer_period = 0.05 if self.mode == 'propeller' else timer_period
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX

        # Create Spacecraft and controller objects
        if self.mode == 'rate':
            from px4_mpc.models.spacecraft_rate_model import SpacecraftRateModel
            from px4_mpc.controllers.spacecraft_rate_mpc import SpacecraftRateMPC
            self.model = SpacecraftRateModel()
            self.mpc = SpacecraftRateMPC(self.model)
        elif self.mode == 'wrench':
            from px4_mpc.models.spacecraft_wrench_model import SpacecraftWrenchModel
            from px4_mpc.controllers.spacecraft_wrench_mpc import SpacecraftWrenchMPC
            self.model = SpacecraftWrenchModel()
            self.mpc = SpacecraftWrenchMPC(self.model)
        elif self.mode == 'offset_free_wrench':
            from px4_mpc.controllers.spacecraft_offset_free_wrench_mpc import SpacecraftOffsetFreeWrenchMPC
            self.mpc = SpacecraftOffsetFreeWrenchMPC()
        elif self.mode == 'lqr_wrench':
            from px4_mpc.models.spacecraft_wrench_model import SpacecraftWrenchModel
            from px4_mpc.controllers.spacecraft_wrench_lqr import SpacecraftWrenchLQR
            self.mpc = SpacecraftWrenchLQR(model=SpacecraftWrenchModel())
        elif self.mode == 'direct_allocation':
            from px4_mpc.models.spacecraft_direct_allocation_model import SpacecraftDirectAllocationModel
            from px4_mpc.controllers.spacecraft_direct_allocation_mpc import SpacecraftDirectAllocationMPC
            self.model = SpacecraftDirectAllocationModel()
            self.mpc = SpacecraftDirectAllocationMPC(self.model)
        elif self.mode == 'propeller':
            from px4_mpc.models.spacecraft_propeller_model import SpacecraftPropellerModel
            from px4_mpc.controllers.spacecraft_propeller_mpc import SpacecraftPropellerMPC
            self.model = SpacecraftPropellerModel()
            self.mpc = SpacecraftPropellerMPC(self.model)
        elif self.mode == 'wrench_cw':
            from px4_mpc.models.spacecraft_wrench_cw_model import SpacecraftWrenchCWModel
            from px4_mpc.controllers.spacecraft_wrench_cw_mpc import SpacecraftWrenchCWMPC
            self.model = SpacecraftWrenchCWModel(orbital_period=self.orbit_period)
            self.mpc = SpacecraftWrenchCWMPC(self.model, skip_build=self.skip_build)

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.setpoint_position = np.array([1.0, 0.0, 0.0])if self.mode != 'wrench_cw' else np.array([0.0, 1.0, 0.0])
        self.setpoint_velocity = np.array([0.0, 0.0, 0.0])
        self.setpoint_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.setpoint_angular_velocity = np.array([0.0, 0.0, 0.0])

        self.trajectory_positions = None  # Will be Nx3 numpy array
        self.trajectory_velocities = None  # Will be Nx3 numpy array
        self.setpoint_trajectory_ready = False

        # Set initial timestamps
        self.vehicle_attitude_timestamp = -np.inf
        self.vehicle_local_position_timestamp = -np.inf
        self.vehicle_angular_velocity_timestamp = -np.inf
        self.vehicle_status_timestamp = -np.inf

    def set_publishers_subscribers(self, qos_profile):
        # Subscribe to both using the same callback
        # - depending on PX4 version, one or the other will be used, but not both
        self.status_sub_v3 = self.create_subscription(
            VehicleStatus,
            'fmu/out/vehicle_status_v3',
            self.vehicle_status_callback,
            qos_profile)
        self.status_sub_v2 = self.create_subscription(
            VehicleStatus,
            'fmu/out/vehicle_status_v2',
            self.vehicle_status_callback,
            qos_profile)
        self.status_sub_v1 = self.create_subscription(
            VehicleStatus,
            'fmu/out/vehicle_status_v1',
            self.vehicle_status_callback,
            qos_profile)
        self.status_sub = self.create_subscription(
            VehicleStatus,
            'fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile)

        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            'fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback,
            qos_profile)
        self.angular_vel_sub = self.create_subscription(
            VehicleAngularVelocity,
            'fmu/out/vehicle_angular_velocity',
            self.vehicle_angular_velocity_callback,
            qos_profile)
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            'fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile)
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            'fmu/out/vehicle_local_position_v1',
            self.vehicle_local_position_callback,
            qos_profile)

        if self.setpoint_from_rviz:
            self.set_pose_srv = self.create_service(
                SetPose,
                'set_pose',
                self.add_set_pose_callback
            )
        else:
            if self.mode == 'wrench_cw':
                self.setpoint_odom_sub = self.create_subscription(
                    Odometry,
                    'px4_mpc/setpoint_odom',
                    self.get_setpoint_odom_callback,
                    0
                )
                self.setpoint_trajectory_sub = self.create_subscription(
                    JointTrajectory,
                    'px4_mpc/setpoint_trajectory',
                    self.get_setpoint_trajectory_callback,
                    0
                )
            self.setpoint_pose_sub = self.create_subscription(
                PoseStamped,
                'px4_mpc/setpoint_pose',
                self.get_setpoint_pose_callback,
                0
            )

        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode,
            'fmu/in/offboard_control_mode',
            qos_profile)
        self.publisher_rates_setpoint = self.create_publisher(
            VehicleRatesSetpoint,
            'fmu/in/vehicle_rates_setpoint',
            qos_profile)
        self.publisher_direct_actuator = self.create_publisher(
            ActuatorMotors,
            'fmu/in/actuator_motors',
            qos_profile)
        self.publisher_thrust_setpoint = self.create_publisher(
            VehicleThrustSetpoint,
            'fmu/in/vehicle_thrust_setpoint',
            qos_profile)
        self.publisher_torque_setpoint = self.create_publisher(
            VehicleTorqueSetpoint,
            'fmu/in/vehicle_torque_setpoint',
            qos_profile)
        self.publisher_propeller_setpoint = self.create_publisher(
            Float32MultiArray,
            'prop_plate/external_motor_cmd',
            10)
        self.predicted_path_pub = self.create_publisher(
            Path,
            'px4_mpc/predicted_path',
            10)
        self.reference_pub = self.create_publisher(
            Marker,
            'px4_mpc/reference',
            10)
        if self.mode == 'offset_free_wrench':
            self.disturbance_rotation_pub = self.create_publisher(
                Vector3Stamped,
                'px4_mpc/translation_d_hat',
                qos_profile)

            self.disturbance_translation_pub = self.create_publisher(
                Vector3Stamped,
                'px4_mpc/attitude_d_hat',
                qos_profile)

        if self.sitl:
            self.odom_pub = self.create_publisher(
                Odometry,
                'odom',
                qos_profile)
        return

    def vehicle_attitude_callback(self, msg):
        # Store message arrival time in ROS clock domain for validity checking
        self.vehicle_attitude_timestamp = self.get_clock().now().nanoseconds / 1e9
        
        if self.use_ned:
            # NED-> ENU transformation
            # Receives quaternion in NED frame as (qw, qx, qy, qz)
            q_enu = 1/np.sqrt(2) * np.array([msg.q[0] + msg.q[3], msg.q[1] + msg.q[2], msg.q[1] - msg.q[2], msg.q[0] - msg.q[3]])
            q_enu /= np.linalg.norm(q_enu)
            self.vehicle_attitude = q_enu.astype(float)
        else:
            q = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]])
            q /= np.linalg.norm(q)
            self.vehicle_attitude = q.astype(float)

    def vehicle_local_position_callback(self, msg):
        # Store message arrival time in ROS clock domain for validity checking
        self.vehicle_local_position_timestamp = self.get_clock().now().nanoseconds / 1e9
        
        if self.use_ned:
            # NED-> ENU transformation
            self.vehicle_local_position[0] = msg.y
            self.vehicle_local_position[1] = msg.x
            self.vehicle_local_position[2] = -msg.z
            self.vehicle_local_velocity[0] = msg.vy
            self.vehicle_local_velocity[1] = msg.vx
            self.vehicle_local_velocity[2] = -msg.vz
        else:
            self.vehicle_local_position[0] = msg.x
            self.vehicle_local_position[1] = msg.y
            self.vehicle_local_position[2] = msg.z
            self.vehicle_local_velocity[0] = msg.vx
            self.vehicle_local_velocity[1] = msg.vy
            self.vehicle_local_velocity[2] = msg.vz

    def vehicle_angular_velocity_callback(self, msg):
        # Store message arrival time in ROS clock domain for validity checking
        self.vehicle_angular_velocity_timestamp = self.get_clock().now().nanoseconds / 1e9
        
        if self.use_ned:
            # NED-> ENU transformation
            self.vehicle_angular_velocity[0] = msg.xyz[0]
            self.vehicle_angular_velocity[1] = -msg.xyz[1]
            self.vehicle_angular_velocity[2] = -msg.xyz[2]
        else:
            self.vehicle_angular_velocity[0] = msg.xyz[0]
            self.vehicle_angular_velocity[1] = msg.xyz[1]
            self.vehicle_angular_velocity[2] = msg.xyz[2]

    def vehicle_status_callback(self, msg):
        # Store message arrival time in ROS clock domain for validity checking
        self.vehicle_status_timestamp = self.get_clock().now().nanoseconds / 1e9
        self.nav_state = msg.nav_state

    def publish_reference(self, pub, reference):
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = "map"
        # msg.header.stamp = Clock().now().nanoseconds / 1000
        msg.ns = "arrow"
        msg.id = 1
        msg.type = Marker.SPHERE
        msg.scale.x = 0.09
        msg.scale.y = 0.09
        msg.scale.z = 0.09
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.pose.position.x = reference[0]
        msg.pose.position.y = reference[1]
        msg.pose.position.z = reference[2]
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0

        pub.publish(msg)

    def publish_rate_setpoint(self, u_pred):
        F_cmd = u_pred[0, 0:3]
        w_cmd = u_pred[0, 3:6]

        # The PX4 uses normalized force input. Scaling with respect to the maximum force.
        F_scaling = 1/(2 * 1.5)
        F_cmd *= F_scaling

        rates_setpoint_msg = VehicleRatesSetpoint()
        rates_setpoint_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        rates_setpoint_msg.roll  = float(w_cmd[0])
        rates_setpoint_msg.pitch = -float(w_cmd[1])
        rates_setpoint_msg.yaw   = -float(w_cmd[2])
        rates_setpoint_msg.thrust_body[0] = float(F_cmd[0])
        rates_setpoint_msg.thrust_body[1] = -float(F_cmd[1])
        rates_setpoint_msg.thrust_body[2] = -float(F_cmd[2])
        self.publisher_rates_setpoint.publish(rates_setpoint_msg)

    def publish_wrench_setpoint(self, u_pred):
        # u_pred is [Fx, Fy, Fz, Tx, Ty, Tz]] in FLU frame
        # The PX4 uses normalized wrench input. Scaling with respect to the maximum force and torque.
        F_scaling = 1/(2 * 1.5)
        T_scaling = 1/(4 * 0.12 * 1.5)
        u_pred[0, :3] *= F_scaling
        u_pred[0, 3:6] *= T_scaling

        timestamp = int(self.get_clock().now().nanoseconds / 1000)

        thrust_outputs_msg = VehicleThrustSetpoint()
        thrust_outputs_msg.timestamp = timestamp

        torque_outputs_msg = VehicleTorqueSetpoint()
        torque_outputs_msg.timestamp = timestamp

        if self.use_ned:
            # FLU -> FRD transformation
            thrust_outputs_msg.xyz = [u_pred[0, 0], -u_pred[0, 1], -u_pred[0, 2]]
            torque_outputs_msg.xyz = [u_pred[0, 3], -u_pred[0, 4], -u_pred[0, 5]]
        else:
            thrust_outputs_msg.xyz = [u_pred[0, 0], u_pred[0, 1], u_pred[0, 2]]
            torque_outputs_msg.xyz = [u_pred[0, 3], u_pred[0, 4], u_pred[0, 5]]

        self.publisher_thrust_setpoint.publish(thrust_outputs_msg)
        self.publisher_torque_setpoint.publish(torque_outputs_msg)

    def publish_direct_actuator_setpoint(self, u_pred):
        actuator_outputs_msg = ActuatorMotors()
        actuator_outputs_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        # Normalize thrust values w.r.t. max thrust
        thrust = u_pred[0, :] / self.model.max_thrust

        # Generate actuator outputs dynamically
        thrust_command = []
        for t in thrust:
            thrust_command.extend([max(t, 0.0), max(-t, 0.0)])
        thrust_command = np.clip(np.array(thrust_command, dtype=np.float32), 0.0, 1.0)

        actuator_outputs_msg.control[:len(thrust_command)] = thrust_command
        self.publisher_direct_actuator.publish(actuator_outputs_msg)

    def publish_propeller_setpoint(self, u_pred):
        min_thrust = -1.5
        max_thrust = 1.5
        propeller_outputs_msg = Float32MultiArray()
        thrust_command = u_pred[0, :]
        thrust_command = np.clip(np.array(thrust_command, dtype=np.float32), min_thrust, max_thrust)
        propeller_outputs_msg.data = thrust_command.tolist()
        self.publisher_propeller_setpoint.publish(propeller_outputs_msg)

    def publish_disturbance_estimate(self, d_hat):
        disturbance_msg = Vector3Stamped()
        disturbance_msg.header.stamp = Clock().now().to_msg()
        disturbance_msg.vector.x = d_hat[0]
        disturbance_msg.vector.y = d_hat[1]
        disturbance_msg.vector.z = d_hat[2]
        self.disturbance_translation_pub.publish(disturbance_msg)

        disturbance_msg = Vector3Stamped()
        disturbance_msg.header.stamp = Clock().now().to_msg()
        disturbance_msg.vector.x = d_hat[3]
        disturbance_msg.vector.y = d_hat[4]
        disturbance_msg.vector.z = d_hat[5]
        self.disturbance_rotation_pub.publish(disturbance_msg)

    def publish_sitl_odometry(self):
        msg = Odometry()
        msg.header.frame_id = "mocap"
        msg.child_frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = self.vehicle_local_position[0]
        msg.pose.pose.position.y = self.vehicle_local_position[1]
        msg.pose.pose.position.z = self.vehicle_local_position[2]
        msg.pose.pose.orientation.w = self.vehicle_attitude[0]
        msg.pose.pose.orientation.x = self.vehicle_attitude[1]
        msg.pose.pose.orientation.y = self.vehicle_attitude[2]
        msg.pose.pose.orientation.z = self.vehicle_attitude[3]
        msg.twist.twist.linear.x = self.vehicle_local_velocity[0]
        msg.twist.twist.linear.y = self.vehicle_local_velocity[1]
        msg.twist.twist.linear.z = self.vehicle_local_velocity[2]
        msg.twist.twist.angular.x = self.vehicle_angular_velocity[0]
        msg.twist.twist.angular.y = self.vehicle_angular_velocity[1]
        msg.twist.twist.angular.z = self.vehicle_angular_velocity[2]
        self.odom_pub.publish(msg)

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "mocap"
        pose_msg.header.stamp = Clock().now().to_msg()
        pose_msg.pose.position.x = self.vehicle_local_position[0]
        pose_msg.pose.position.y = self.vehicle_local_position[1]
        pose_msg.pose.position.z = self.vehicle_local_position[2]
        pose_msg.pose.orientation.w = self.vehicle_attitude[0]
        pose_msg.pose.orientation.x = self.vehicle_attitude[1]
        pose_msg.pose.orientation.y = self.vehicle_attitude[2]
        pose_msg.pose.orientation.z = self.vehicle_attitude[3]
        self.sitl_pose_pub.publish(pose_msg)
        return

    def check_data_validity(self):
        ret_val = True
        current_time = self.get_clock().now().nanoseconds / 1e9

        # Check if the data is valid based on the timestamps
        if (current_time - self.vehicle_attitude_timestamp > DATA_VALIDITY_STREAM):
            self.get_logger().warn("Vehicle attitude data is too old. Skipping offboard control...")
            ret_val = False
        if (current_time - self.vehicle_local_position_timestamp > DATA_VALIDITY_STREAM):
            self.get_logger().warn("Vehicle position data is too old. Skipping offboard control...")
            ret_val = False
        if (current_time - self.vehicle_angular_velocity_timestamp > DATA_VALIDITY_STREAM):
            self.get_logger().warn("Vehicle angular velocity data is too old. Skipping offboard control...")
            ret_val = False
        if (current_time - self.vehicle_status_timestamp > DATA_VALIDITY_STATUS):
            self.get_logger().warn("Vehicle status data is too old. Skipping offboard control...")
            ret_val = False
        return ret_val

    def cmdloop_callback(self):

        # Publish odometry for SITL
        if self.sitl:
            self.publish_sitl_odometry()

        # Check data validity
        if not self.check_data_validity():
            return

        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.direct_actuator = False
        if self.mode == 'rate':
            offboard_msg.body_rate = True
        elif self.mode == 'direct_allocation' or self.mode == 'propeller':
            offboard_msg.direct_actuator = True
        elif self.mode == 'wrench' or self.mode == 'offset_free_wrench' or self.mode == 'lqr_wrench' or self.mode == 'wrench_cw':
            offboard_msg.thrust_and_torque = True
        if not self.mode == 'wrench_cw':
            self.publisher_offboard_mode.publish(offboard_msg)

        # Set state and references for each MPC
        if self.mode == 'rate':
            x0 = np.array([self.vehicle_local_position[0],
                           self.vehicle_local_position[1],
                           self.vehicle_local_position[2],
                           self.vehicle_local_velocity[0],
                           self.vehicle_local_velocity[1],
                           self.vehicle_local_velocity[2],
                           self.vehicle_attitude[0],
                           self.vehicle_attitude[1],
                           self.vehicle_attitude[2],
                           self.vehicle_attitude[3]]).reshape(10, 1)
            ref = np.concatenate((self.setpoint_position,       # position
                                  np.zeros(3),                  # velocity
                                  self.setpoint_attitude,       # attitude
                                  np.zeros(6)), axis=0)         # inputs reference (F, w)
            ref = np.repeat(ref.reshape((-1, 1)), self.mpc.N + 1, axis=1)
        elif self.mode == 'wrench' or self.mode == 'offset_free_wrench':
            x0 = np.array([self.vehicle_local_position[0],
                           self.vehicle_local_position[1],
                           self.vehicle_local_position[2],
                           self.vehicle_local_velocity[0],
                           self.vehicle_local_velocity[1],
                           self.vehicle_local_velocity[2],
                           self.vehicle_attitude[0],
                           self.vehicle_attitude[1],
                           self.vehicle_attitude[2],
                           self.vehicle_attitude[3],
                           self.vehicle_angular_velocity[0],
                           self.vehicle_angular_velocity[1],
                           self.vehicle_angular_velocity[2]]).reshape(13, 1)
            ref = np.concatenate((self.setpoint_position,       # position
                                  np.zeros(3),                  # velocity
                                  self.setpoint_attitude,       # attitude
                                  np.zeros(3),                  # angular velocity
                                  np.zeros(6)), axis=0)         # inputs reference (F, torque)
            ref = np.repeat(ref.reshape((-1, 1)), self.mpc.N + 1, axis=1)
        elif self.mode == 'lqr_wrench':
            x0 = np.array([self.vehicle_local_position[0],
                           self.vehicle_local_position[1],
                           self.vehicle_local_position[2],
                           self.vehicle_local_velocity[0],
                           self.vehicle_local_velocity[1],
                           self.vehicle_local_velocity[2],
                           self.vehicle_attitude[0],
                           self.vehicle_attitude[1],
                           self.vehicle_attitude[2],
                           self.vehicle_attitude[3],
                           self.vehicle_angular_velocity[0],
                           self.vehicle_angular_velocity[1],
                           self.vehicle_angular_velocity[2]]).reshape(13, 1)
            ref = np.concatenate((self.setpoint_position,       # position
                                  np.zeros(3),                  # velocity
                                  self.setpoint_attitude[0:],       # attitude
                                  np.zeros(3)), axis=0)         # angular velocity
        elif self.mode == 'direct_allocation' or self.mode == 'propeller':
            x0 = np.array([self.vehicle_local_position[0],
                           self.vehicle_local_position[1],
                           self.vehicle_local_position[2],
                           self.vehicle_local_velocity[0],
                           self.vehicle_local_velocity[1],
                           self.vehicle_local_velocity[2],
                           self.vehicle_attitude[0],
                           self.vehicle_attitude[1],
                           self.vehicle_attitude[2],
                           self.vehicle_attitude[3],
                           self.vehicle_angular_velocity[0],
                           self.vehicle_angular_velocity[1],
                           self.vehicle_angular_velocity[2]]).reshape(13, 1)
            ref = np.concatenate((self.setpoint_position,       # position
                                  np.zeros(3),                  # velocity
                                  self.setpoint_attitude,       # attitude
                                  np.zeros(3),                  # angular velocity
                                  np.zeros(4)), axis=0)         # inputs reference (u1, ..., u4) for 2D platform
            ref = np.repeat(ref.reshape((-1, 1)), self.mpc.N + 1, axis=1)
        elif self.mode == 'wrench_cw':
            # Rotate lab frame -90 deg so y (orbit velocity vector) points towards middle of lab space
            pos_hill = np.array([-self.vehicle_local_position[1], self.vehicle_local_position[0], self.vehicle_local_position[2]])
            vel_hill = np.array([-self.vehicle_local_velocity[1], self.vehicle_local_velocity[0], self.vehicle_local_velocity[2]])
            q_enu = self.vehicle_attitude
            q_hill = np.array([q_enu[0] * np.cos(np.pi/4) - q_enu[3] * np.sin(np.pi/4),
                                q_enu[1] * np.cos(np.pi/4) - q_enu[2] * np.sin(np.pi/4),
                                q_enu[1] * np.sin(np.pi/4) + q_enu[2] * np.cos(np.pi/4),
                                q_enu[0] * np.sin(np.pi/4) + q_enu[3] * np.cos(np.pi/4)])

            x0 = np.array([pos_hill[0],
                            pos_hill[1],
                            pos_hill[2],
                            vel_hill[0],
                            vel_hill[1],
                            vel_hill[2],
                            q_hill[0],
                            q_hill[1],
                            q_hill[2],
                            q_hill[3],
                            self.vehicle_angular_velocity[0],
                            self.vehicle_angular_velocity[1],
                            self.vehicle_angular_velocity[2]]).reshape(13, 1)

            if self.trajectory_positions is None:
                return

            if self.setpoint_from_rviz:
                setpoint_pos_hill = np.array([-self.setpoint_position[1], self.setpoint_position[0], self.setpoint_position[2]])
                setpoint_q_enu = self.setpoint_attitude
                setpoint_q_hill = np.array([setpoint_q_enu[0] * np.cos(np.pi/4) - setpoint_q_enu[3] * np.sin(np.pi/4),
                                            setpoint_q_enu[1] * np.cos(np.pi/4) - setpoint_q_enu[2] * np.sin(np.pi/4),
                                            setpoint_q_enu[1] * np.sin(np.pi/4) + setpoint_q_enu[2] * np.cos(np.pi/4),
                                            setpoint_q_enu[0] * np.sin(np.pi/4) + setpoint_q_enu[3] * np.cos(np.pi/4)])
                ref = np.concatenate((setpoint_pos_hill,        # position
                                        np.zeros(3),            # velocity
                                        setpoint_q_hill,        # attitude
                                        np.zeros(3),            # angular velocity
                                        np.zeros(6)), axis=0)   # inputs reference
                ref = np.repeat(ref.reshape((-1, 1)), self.mpc.N + 1, axis=1)

            else:
                # Forward propagate attitude from planner's odom
                quats, ang_vels = self.forward_propagate_attitude(
                    self.setpoint_attitude,
                    self.setpoint_angular_velocity[2],
                    self.mpc.Tf / self.mpc.N,  # MPC timestep
                    self.mpc.N
                )

                N_horizon = self.mpc.N + 1
                ref = np.zeros((19, N_horizon))
                for i in range(N_horizon):
                    if self.setpoint_trajectory_ready and i < len(self.trajectory_positions):
                        ref[0:3, i] = self.trajectory_positions[i]
                        ref[3:6, i] = self.trajectory_velocities[i]
                    else:
                        ref[0:3, i] = self.setpoint_position
                        ref[3:6, i] = self.setpoint_velocity
                    ref[6:10, i] = quats[i]
                    ref[10:13, i] = ang_vels[i]
                    ref[13:19, i] = 0.0  # input reference
        else:
            raise ValueError(f'Invalid mode: {self.mode}')

        # Solve MPC
        u_pred, x_pred = self.mpc.solve(x0, ref=ref)

        if self.mode == 'offset_free_wrench':
            # Publish disturbance
            self.publish_disturbance_estimate(self.mpc.get_disturbance_estimate())

        # Colect data
        idx = 0
        predicted_path_msg = Path()
        for predicted_state in x_pred:
            idx = idx + 1
            # Publish time history of the vehicle path
            predicted_pose_msg = self.vector2PoseMsg('map', predicted_state[0:3], self.setpoint_attitude)
            predicted_path_msg.header = predicted_pose_msg.header
            predicted_path_msg.poses.append(predicted_pose_msg)
        self.predicted_path_pub.publish(predicted_path_msg)
        self.publish_reference(self.reference_pub, self.setpoint_position)

        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if self.mode == 'rate':
                self.publish_rate_setpoint(u_pred)
            elif self.mode == 'direct_allocation' or self.mode == 'direct_allocation_trajectory':
                self.publish_direct_actuator_setpoint(u_pred)
            elif self.mode == 'wrench' or self.mode == 'offset_free_wrench' or self.mode == 'lqr_wrench' or self.mode == 'wrench_cw':
                 self.publish_wrench_setpoint(u_pred)
            elif self.mode == 'propeller':
                self.publish_propeller_setpoint(u_pred)
                self.publish_wrench_setpoint(np.array([[0.0]*6]*self.mpc.N))  # Publish zero wrench setpoint since control is done via propeller setpoint

    def forward_propagate_attitude(self, q0, omega_z, dt, N):
        """Propagate yaw-only attitude over N steps.
        
        Returns (N+1) x 4 array of quaternions and (N+1) x 3 array of angular velocities.
        Angular rate is constant (no torque prediction in the planner horizon).
        """
        quats = np.zeros((N + 1, 4))
        ang_vels = np.zeros((N + 1, 3))
        quats[0] = q0
        ang_vels[0] = [0.0, 0.0, 0]

        for i in range(1, N + 1):
            dtheta = omega_z * dt
            w0, x0, y0, z0 = quats[i - 1]
            c, s = np.cos(dtheta / 2), np.sin(dtheta / 2)
            quats[i] = np.array([
                w0 * c - z0 * s,
                x0 * c + y0 * s,
            -x0 * s + y0 * c,
                w0 * s + z0 * c,
            ])
            # quats[i] /= np.linalg.norm(quats[i])
            # ang_vels[i] = [0.0, 0.0, omega_z]
            quats[i] = [1.0, 0.0, 0.0, 0.0]
            ang_vels[i] = [0.0, 0.0, 0.0]

        return quats, ang_vels

    def add_set_pose_callback(self, request, response):
        self.setpoint_position[0] = request.pose.position.x
        self.setpoint_position[1] = request.pose.position.y
        self.setpoint_position[2] = request.pose.position.z
        self.setpoint_attitude[0] = request.pose.orientation.w
        self.setpoint_attitude[1] = request.pose.orientation.x
        self.setpoint_attitude[2] = request.pose.orientation.y
        self.setpoint_attitude[3] = request.pose.orientation.z
        return response

    def get_setpoint_pose_callback(self, msg):
        self.setpoint_position[0] = msg.pose.position.x
        self.setpoint_position[1] = msg.pose.position.y
        self.setpoint_position[2] = msg.pose.position.z
        self.setpoint_attitude[0] = msg.pose.orientation.w
        self.setpoint_attitude[1] = msg.pose.orientation.x
        self.setpoint_attitude[2] = msg.pose.orientation.y
        self.setpoint_attitude[3] = msg.pose.orientation.z

    def get_setpoint_odom_callback(self, msg: Odometry):
        self.setpoint_position[0] = msg.pose.pose.position.x
        self.setpoint_position[1] = msg.pose.pose.position.y
        self.setpoint_position[2] = msg.pose.pose.position.z
        self.setpoint_velocity[0] = msg.twist.twist.linear.x
        self.setpoint_velocity[1] = msg.twist.twist.linear.y
        self.setpoint_velocity[2] = msg.twist.twist.linear.z
        self.setpoint_attitude[0] = msg.pose.pose.orientation.w
        self.setpoint_attitude[1] = msg.pose.pose.orientation.x
        self.setpoint_attitude[2] = msg.pose.pose.orientation.y
        self.setpoint_attitude[3] = msg.pose.pose.orientation.z
        self.setpoint_angular_velocity[0] = msg.twist.twist.angular.x
        self.setpoint_angular_velocity[1] = msg.twist.twist.angular.y
        self.setpoint_angular_velocity[2] = msg.twist.twist.angular.z

    def get_setpoint_trajectory_callback(self, msg: JointTrajectory):
        if not msg.points:
            self.get_logger().warn("Received empty trajectory message. Ignoring...")
            return
        N = len(msg.points)
        positions = np.zeros((N, 3))
        velocities = np.zeros((N, 3))
        for i, point in enumerate(msg.points):
            positions[i] = point.positions[:3]
            velocities[i] = point.velocities[:3]
        self.trajectory_positions = positions
        self.trajectory_velocities = velocities
        self.setpoint_trajectory_ready = True

    def vector2PoseMsg(self, frame_id, position, attitude):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.orientation.w = attitude[0]
        pose_msg.pose.orientation.x = attitude[1]
        pose_msg.pose.orientation.y = attitude[2]
        pose_msg.pose.orientation.z = attitude[3]
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2]) if not self.camera else 0.0
        return pose_msg


def main(args=None):
    rclpy.init(args=args)

    spacecraft_mpc = SpacecraftMPC()

    rclpy.spin(spacecraft_mpc)

    spacecraft_mpc.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
