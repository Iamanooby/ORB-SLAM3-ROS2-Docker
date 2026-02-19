import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class PclToCsv(Node):

    def __init__(self):
        super().__init__('pcl_to_csv')
        self.subscription = self.create_subscription(
            PointCloud2,
            # 'global_pointcloud',
            "map_points",
            # "lidar/points",
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # 1. Convert ROS2 PointCloud2 to structured NumPy array
        # This handles the byte-unpacking for you automatically
        data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        
        # 2. Convert to a standard (N, 3) float array
        points_np = np.array(list(data))

        print(points_np)

        if points_np.size > 0:
            # 3. Save directly to CSV using NumPy
            # fmt='%.6f' controls precision (6 decimal places)
            # comments='' prevents NumPy from adding a '#' before the header
            np.savetxt(
                "orb_map_points.csv", 
                points_np, 
                delimiter=",", 
                header="x,y,z", 
                comments="", 
                fmt="%.6f"
            )
            
            self.get_logger().info(f'Saved {len(points_np)} points to orb_map_points.csv')
            rclpy.shutdown()             



def main(args=None):
    rclpy.init(args=args)

    pcl_to_csv = PclToCsv()

    rclpy.spin(pcl_to_csv)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pcl_to_csv.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()