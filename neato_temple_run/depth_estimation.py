import time
import rclpy
from threading import Thread
from rclpy.node import Node
import time
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from matplotlib import pyplot as plt
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import LaserScan

from vision_pipeline.depth_estimation_single_photo import cv2_wrapper
from vision_pipeline.create_obstacles import (
    remove_depth_floor,
    identify_obstacles,
    generate_point_cloud,
)


class DepthEstimation(Node):
    """The DepthEstimation is a Python object that encompasses a ROS node
    that can process images from the camera."""

    def __init__(self, image_topic):
        """Initialize depth_estimation"""
        super().__init__("depth_estimation")
        self.cv_image = None  # the latest image from the camera
        self.bridge = CvBridge()  # used to convert ROS messages to OpenCV

        self.cartesian_points_vision = np.array([])
        self.cartesian_points_lidar = np.array([])

        self.create_subscription(Image, image_topic, self.process_image, 10)
        self.create_subscription(LaserScan, "scan", self.process_scan, 10)

        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.monocular_depth_pub = self.create_publisher(
            PointCloud, "monocular_depth", 10
        )
        self.lidar_depth_pub = self.create_publisher(PointCloud, "lidar_depth", 10)

        self.timer = self.create_timer(0.1, self.publish_monocular_depth_estimates)
        self.timer1 = self.create_timer(0.1, self.publish_lidar_depth_estimates)

        thread = Thread(target=self.loop_wrapper)
        thread.start()

    def process_image(self, msg):
        """Process image messages from ROS and stash them in an attribute
        called cv_image for subsequent processing"""
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def process_scan(self, msg: LaserScan):
        """
        Callback for LaserScan
        Changes the value of self.angular_vel according to the detected wall
        """
        distances = np.array(msg.ranges)
        angles = np.array(range(361))
        x = distances * np.cos(angles * np.pi / 180)
        y = distances * np.sin(angles * np.pi / 180)
        self.cartesian_points_lidar = np.concatenate((x[:, None], y[:, None]), axis=1)

    def loop_wrapper(self):
        """This function takes care of calling the run_loop function repeatedly.
        We are using a separate thread to run the loop_wrapper to work around
        issues with single threaded executors in ROS2"""
        while True:
            self.run_loop()
            time.sleep(0.05)  # might not be long enough

    def run_loop(self):
        # NOTE: only do cv2.imshow and cv2.waitKey in this function
        if not self.cv_image is None:
            depth = cv2_wrapper(self.cv_image, 64)
            depth_remove_floor = remove_depth_floor(depth)
            masked_map, mask = identify_obstacles(depth_remove_floor, depth)
            self.cartesian_points_vision, width = generate_point_cloud(depth, mask)
            if not self.cartesian_points_vision:
                cp = [[0, -100, -100], [0, -200, -200]]
            else:
                cp = self.cartesian_points_vision
            # plt.imshow(
            #     depth,
            #     cmap=plt.get_cmap("hot"),
            #     interpolation="nearest",
            #     vmin=0,
            #     vmax=1,
            # )

            plt.scatter(cp[:, 0], cp[:, 2])
            plt.xlim(-90, 90)
            plt.ylim(0, 130)

            plt.show(block=False)
            plt.pause(0.05)
            plt.clf()

    def publish_monocular_depth_estimates(self):
        msg = PointCloud()
        msg.header.stamp = Time()
        msg.header.frame_id = "odom"
        msg.points = [Point32(x=p[0], y=p[2]) for p in self.cartesian_points_vision]
        self.monocular_depth_pub.publish(msg)

    def publish_lidar_depth_estimates(self):
        msg = PointCloud()
        msg.header.stamp = Time()
        msg.header.frame_id = "odom"
        msg.points = [Point32(x=p[0], y=p[1]) for p in self.cartesian_points_lidar]
        self.monocular_depth_pub.publish(msg)


if __name__ == "__main__":
    node = DepthEstimation("/camera/image_raw")
    node.run()


def main(args=None):
    rclpy.init()
    n = DepthEstimation("camera/image_raw")
    rclpy.spin(n)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
