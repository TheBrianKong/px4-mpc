#!/usr/bin/env python3

"""
Fixed-Wing Visualization and Diagnostics Node
Subscribes to raw state and MPC predictions to publish RViz Markers.
"""

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster

from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition

class FixedWingVizNode(Node):
    def __init__(self):
        super().__init__('fw_viz_node')
        
        qos_profile_sub = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, 
                                durability=QoSDurabilityPolicy.VOLATILE, 
                                history=QoSHistoryPolicy.KEEP_LAST, depth=1)

        # sub directly to px4
        self.attitude_sub = self.create_subscription(VehicleAttitude, 
            '/fmu/out/vehicle_attitude', self.attitude_callback, qos_profile_sub)
        self.local_position_sub = self.create_subscription(VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1', self.position_callback, qos_profile_sub)
            
        # sub to mpc node
        self.optimal_path_sub = self.create_subscription(Float64MultiArray,
            '/mpc/traj_optimal_raw', self.optimal_path_callback, 1)

        # publish for rviz
        self.vehicle_path_pub = self.create_publisher(Path, "px4_viz/vehicle_path", 10)
        self.vehicle_pose_pub = self.create_publisher(MarkerArray, "px4_viz/vehicle_pose", 10)
        self.optimal_path_pub = self.create_publisher(MarkerArray, "px4_viz/optimal_path", 1)
        self.reference_path_pub = self.create_publisher(Marker, "px4_viz/reference_path", 1)
        self.tf_broadcaster = TransformBroadcaster(self)

        # internal vars for plotting
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_path_msg = Path()
        self.trail_size = 2048
        
        # TODO: this should not be hard coded
        self.path_center = np.array([30.0, 15.0, 30.0])
        self.path_radius = 50.0

        # Publish visualizations at 50Hz
        self.timer = self.create_timer(1.0 / 50.0, self.publish_visuals)

    # subscriber callbacks
    def attitude_callback(self, msg):
        q_enu = 1/np.sqrt(2) * np.array([msg.q[0] + msg.q[3], msg.q[1] + msg.q[2], msg.q[1] - msg.q[2], msg.q[0] - msg.q[3]])
        self.vehicle_attitude[:] = q_enu / np.linalg.norm(q_enu)
        
    def position_callback(self, msg):
        self.vehicle_local_position[:] = [msg.y, msg.x, -msg.z]

    def optimal_path_callback(self, msg):
        """Receives flattened 1D array of x_pred from MPC and reconstructs triads"""
        if not msg.data: return
        
        data = np.array(msg.data)
        nx = 8 # Ensure this matches your MPC state dimension
        horizon = len(data) // nx
        x_pred = data.reshape((horizon, nx))
        
        self.publish_optimal_path(x_pred)

    # all the visual publishers
    def publish_visuals(self):
        # tf broadcaster
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = float(self.vehicle_local_position[0])
        t.transform.translation.y = float(self.vehicle_local_position[1])
        t.transform.translation.z = float(self.vehicle_local_position[2])
        t.transform.rotation.w = float(self.vehicle_attitude[0])
        t.transform.rotation.x = float(self.vehicle_attitude[1])
        t.transform.rotation.y = float(self.vehicle_attitude[2])
        t.transform.rotation.z = float(self.vehicle_attitude[3])
        self.tf_broadcaster.sendTransform(t)

        # trail msg
        pose_msg = PoseStamped()
        pose_msg.header = t.header
        pose_msg.pose.orientation = t.transform.rotation
        pose_msg.pose.position.x = t.transform.translation.x
        pose_msg.pose.position.y = t.transform.translation.y
        pose_msg.pose.position.z = t.transform.translation.z
        
        self.vehicle_path_msg.header = pose_msg.header
        self.vehicle_path_msg.poses.append(pose_msg)
        if len(self.vehicle_path_msg.poses) > self.trail_size:
            del self.vehicle_path_msg.poses[0]
        self.vehicle_path_pub.publish(self.vehicle_path_msg)

        # 3. Vehicle Mesh & Static Circle
        self.publish_vehicle_pose()
        self.publish_reference_path()

    def publish_vehicle_pose(self):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        
        mesh_rotation = np.array([np.cos(np.pi/2), 0.0, 0.0, np.sin(np.pi/2)]) 
        mesh_attitude = self.quat_multiply(self.vehicle_attitude, mesh_rotation)
        
        mesh_marker = Marker()
        mesh_marker.header.stamp = stamp
        mesh_marker.header.frame_id = "map"
        mesh_marker.ns = "vehicle_mesh"
        mesh_marker.id = 0
        mesh_marker.type = Marker.MESH_RESOURCE
        mesh_marker.mesh_resource = "package://px4_mpc/resource/believer.dae" 
        mesh_marker.scale.x = mesh_marker.scale.y = mesh_marker.scale.z = 1.0
        mesh_marker.color.a = 0.8  
        mesh_marker.color.r = mesh_marker.color.g = mesh_marker.color.b = 0.7  
        mesh_marker.pose.position.x = float(self.vehicle_local_position[0])
        mesh_marker.pose.position.y = float(self.vehicle_local_position[1])
        mesh_marker.pose.position.z = float(self.vehicle_local_position[2])
        mesh_marker.pose.orientation.w = float(mesh_attitude[0])
        mesh_marker.pose.orientation.x = float(mesh_attitude[1])
        mesh_marker.pose.orientation.y = float(mesh_attitude[2])
        mesh_marker.pose.orientation.z = float(mesh_attitude[3])
        mesh_marker.action = Marker.ADD
        marker_array.markers.append(mesh_marker)
        
        def create_axis_marker(m_id, r, g, b):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = "map"
            m.ns = "vehicle_axes"
            m.id = m_id
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.08 
            m.color.a = 1.0
            m.color.r, m.color.g, m.color.b = float(r), float(g), float(b)
            return m

        # Safe unique IDs for the vehicle axes
        marker_x = create_axis_marker(10, 1.0, 0.0, 0.0) 
        marker_y = create_axis_marker(11, 0.0, 1.0, 0.0) 
        marker_z = create_axis_marker(12, 0.0, 0.0, 1.0) 

        axis_length = 5.0 
        R_mat = self.quat_to_rot_matrix(self.vehicle_attitude)
        pos = self.vehicle_local_position
        p_center = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        
        p_x = Point(x=float(pos[0] + R_mat[0, 0] * axis_length), y=float(pos[1] + R_mat[1, 0] * axis_length), z=float(pos[2] + R_mat[2, 0] * axis_length))
        p_y = Point(x=float(pos[0] + R_mat[0, 1] * axis_length), y=float(pos[1] + R_mat[1, 1] * axis_length), z=float(pos[2] + R_mat[2, 1] * axis_length))
        p_z = Point(x=float(pos[0] + R_mat[0, 2] * axis_length), y=float(pos[1] + R_mat[1, 2] * axis_length), z=float(pos[2] + R_mat[2, 2] * axis_length))
        
        marker_x.points.extend([p_center, p_x])
        marker_y.points.extend([p_center, p_y])
        marker_z.points.extend([p_center, p_z])
        marker_array.markers.extend([marker_x, marker_y, marker_z])
        
        self.vehicle_pose_pub.publish(marker_array)

    def publish_optimal_path(self, optimal_trajectory):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        
        def create_triad_marker(marker_id, r, g, b):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = "map"
            m.ns = "optimal_path_triads"
            m.id = marker_id
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.25
            m.color.a = 0.8
            m.color.r, m.color.g, m.color.b = float(r), float(g), float(b)
            return m

        marker_x = create_triad_marker(1, 1.0, 0.0, 0.0) 
        marker_y = create_triad_marker(2, 0.0, 1.0, 0.0) 
        marker_z = create_triad_marker(3, 0.0, 0.0, 1.0) 

        axis_length = 2
        qs = optimal_trajectory[:,4:8]
        safe_norms = np.where(np.linalg.norm(qs, axis=1, keepdims=True) > 1e-6, np.linalg.norm(qs, axis=1, keepdims=True), 1.0)
        qs_norm = qs / safe_norms
        
        for t in range(0, optimal_trajectory.shape[0], 5):
            pos = optimal_trajectory[t, 0:3]
            R_mat = self.quat_to_rot_matrix(qs_norm[t])
            p_center = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            
            p_x = Point(x=float(pos[0] + R_mat[0, 0] * axis_length), y=float(pos[1] + R_mat[1, 0] * axis_length), z=float(pos[2] + R_mat[2, 0] * axis_length))
            p_y = Point(x=float(pos[0] + R_mat[0, 1] * axis_length), y=float(pos[1] + R_mat[1, 1] * axis_length), z=float(pos[2] + R_mat[2, 1] * axis_length))
            p_z = Point(x=float(pos[0] + R_mat[0, 2] * axis_length), y=float(pos[1] + R_mat[1, 2] * axis_length), z=float(pos[2] + R_mat[2, 2] * axis_length))
            
            marker_x.points.extend([p_center, p_x])
            marker_y.points.extend([p_center, p_y])
            marker_z.points.extend([p_center, p_z])
            
        marker_array.markers.extend([marker_x, marker_y, marker_z])
        self.optimal_path_pub.publish(marker_array)

    def publish_reference_path(self):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = "reference_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.3
        marker.color.a = 0.9
        marker.color.r = 1.0; marker.color.b = 1.0
        
        for theta in np.linspace(0.0, 2.0 * np.pi, 50):
            marker.points.append(Point(
                x=float(self.path_center[0] + self.path_radius * np.cos(theta)),
                y=float(self.path_center[1] + self.path_radius * np.sin(theta)),
                z=float(self.path_center[2])
            ))
        self.reference_path_pub.publish(marker)
    # maybe i should've imported these? not worth the time thinking about it.
    def quat_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])

    def quat_to_rot_matrix(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

def main(args=None):
    rclpy.init(args=args)
    node = FixedWingVizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()