#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from slam_msgs.srv import GetLandmarksInView
import time

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class PoseSubscriberNode(Node):
    def __init__(self):
        super().__init__('pose_subscriber_node')
        
        # Subscribe to a PoseStamped topic (adjust the topic name as needed)
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            # '/robot_pose_slam',  # Replace with your actual pose topic
            '/camera_pose',
            self.pose_callback,
            10  # QoS profile, adjust if necessary
        )
        
        # Create a service client for GetLandmarksInView
        self.client = self.create_client(GetLandmarksInView, 'orb_slam3/get_landmarks_in_view')

        # Ensure that the service is available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

        self.get_logger().info('Pose Subscriber Node started and waiting for messages.')


        self.cam_pose_publisher = self.create_publisher(
            PoseStamped,
            "/camera_pose",
            10
        )

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


    def pose_callback(self, msg: PoseStamped):
        # Extract pose from the PoseStamped message
        pose = msg.pose
        self.get_logger().info(f'Received pose: {pose}')

        # Create a service request
        request = GetLandmarksInView.Request()
        request.pose = pose  # Set the pose from the message into the service request
        request.max_angle_pose_observation = 4.0 # get all kfs visible from any angle.
        request.max_dist_pose_observation = 12.0

        self.start_time = time.time()
        # Send the service request asynchronously
        future = self.client.call_async(request)
        future.add_done_callback(self.service_response_callback)

    def service_response_callback(self, future):
        try:
            # Get the result from the service call
            response = future.result()
            end_time = time.time()
            total_time = end_time - self.start_time
            
            if response:
                self.get_logger().info(f'Received {len(response.map_points)} map points from the service.')
            else:
                self.get_logger().info('No map points received from the service.')
            
            self.get_logger().info(f'Time taken for service call: {total_time:.4f} seconds')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')


    def timer_callback(self):
        from_frame_rel = 'map'
        to_frame_rel = 'camera_link'
        try:
            t = self.tf_buffer.lookup_transform(
                from_frame_rel,
                to_frame_rel,
                rclpy.time.Time())
            self.get_logger().info(f'Received pose: {t}')
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return


        msg = PoseStamped()
        msg.header = t.header
        msg.pose.position.x = t.transform.translation.x
        msg.pose.position.y = t.transform.translation.y
        msg.pose.position.z = t.transform.translation.z
        msg.pose.orientation = t.transform.rotation

        # self.get_logger().info(f'Received pose: {msg.pose}')

        self.cam_pose_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PoseSubscriberNode()
    
    try:
        # Spin the node so the callback is called
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Shutdown the node
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
