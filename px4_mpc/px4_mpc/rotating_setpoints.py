#!/usr/bin/env python3
import sys
import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32

from px4_msgs.msg import VehicleLocalPosition
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

class RotatingSetpointPublisher(Node):
    def __init__(self):
        super().__init__('rotating_setpoint_publisher')

        self.namespace = self.declare_parameter('namespace', '').value
        self.namespace_prefix = f'/{self.namespace}/' if self.namespace else ''
        
        self.publisher_ = self.create_publisher(PoseStamped, f'{self.namespace_prefix}px4_mpc/setpoint_pose', 10)
        self.command_sub = self.create_subscription(Int32, f'{self.namespace_prefix}px4_mpc/command', self.command_callback, 10)

        # Rotation parameters
        self.center_x = 2.0
        self.center_y = 0.0
        self.center_z = 0.0
        self.rotation_speed = 0.1  # Rad/s

        self.is_running = False
        self.has_received_pos = False
        self.current_yaw = 0.0

        self.setpoint_hz = 10 # Publish Setpoint Frequency
        self.timer_period = 1/self.setpoint_hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Update Rotation Center
        self.declare_parameter('update_rotation_center', False)
        self.update_center = self.get_parameter('update_rotation_center').value
        self.get_logger().info(f"aSDUHSDAJI {self.update_center}")

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0
        )
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            'fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile_sub)
            
        self.local_position_sub_v1 = self.create_subscription(
            VehicleLocalPosition,
            'fmu/out/vehicle_local_position_v1',
            self.vehicle_local_position_callback,
            qos_profile_sub)
        
        self.get_logger().info(f'Waiting for command on {self.command_sub.topic} topic...')
    
    def vehicle_local_position_callback(self, msg):
        # NED -> ENU transform
        self.current_robot_x = msg.y
        self.current_robot_y = msg.x
        self.current_robot_z = -msg.z
        self.has_received_pos = True

    def command_callback(self, msg):
        command = msg.data
        if command == 1:
            if not self.is_running:
                if self.update_center and self.has_received_pos:
                    self.center_x = self.current_robot_x
                    self.center_y = self.current_robot_y
                    self.center_z = self.current_robot_z
                    self.get_logger().info(f'Updating rotation center: ({self.center_x:.2f}, {self.center_y:.2f}, {self.center_z:.2f})')
                else:
                    self.get_logger().warn('No local position received yet! Using default center')
                self.get_logger().info('Starting rotation...')
                self.is_running = True
        elif command == 0:
            if self.is_running:
                self.get_logger().info('Stopping rotation...')
                self.is_running = False
       
    def timer_callback(self):
        if self.is_running:
            self.current_yaw += self.rotation_speed * self.timer_period
            
            if self.current_yaw > math.pi:
                self.current_yaw -= 2 * math.pi

            # Yaw to Quat
            qx = 0.0
            qy = 0.0
            qz = math.sin(self.current_yaw / 2.0)
            qw = math.cos(self.current_yaw / 2.0)

            # Publish msg
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()

            pose.pose.position.x = self.center_x
            pose.pose.position.y = self.center_y
            pose.pose.position.z = self.center_z
            
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            
            self.publisher_.publish(pose)

def main(args=None):
    if args is None:
        args = sys.argv

    rclpy.init(args=args)
    node = RotatingSetpointPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('\nShutting down node.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()