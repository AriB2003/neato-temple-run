#!/usr/bin/env python3

import time
import rclpy
from threading import Thread
from rclpy.node import Node
import time
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from matplotlib import pyplot as plt
from sensor_msgs.msg import PointCloud, LaserScan

# need to be changed when importing
from depth_estimation_single_photo import cv2_wrapper
from create_obstacles import (
    remove_depth_floor,
    identify_obstacles,
    generate_point_cloud,
)
from angle_helpers import euler_from_quaternion, convert_to_odom


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
        self.x = 0
        self.y = 0
        self.theta = 0

        self.create_subscription(Odometry, "odom", self.process_odom, 10)
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

    def process_odom(self, msg: Odometry):
        """
        Callback for odometry,
        - puts self.odom into the form (x,y,theta)
        """
        orientation_tuple = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.theta = euler_from_quaternion(*orientation_tuple)[2]

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
        valid_distances_bool = tuple([distances != np.inf])
        distances = distances[valid_distances_bool]
        angles = angles[valid_distances_bool]

        x = distances * np.cos(angles * np.pi / 180)
        y = distances * np.sin(angles * np.pi / 180)

        self.cartesian_points_lidar = np.concatenate((x[:, None], y[:, None]), axis=1)

        self.cartesian_points_lidar = convert_to_odom(
            self.cartesian_points_lidar, self.x, self.y, self.theta
        )

    def loop_wrapper(self):
        """This function takes care of calling the run_loop function repeatedly.
        We are using a separate thread to run the loop_wrapper to work around
        issues with single threaded executors in ROS2"""
        while True:
            self.run_loop()
            time.sleep(0.05)  # might not be long enough

    def run_loop(self):
        """
        Obtains cartesian points from vision,
        Plots points of interest
        """
        # NOTE: only do cv2.imshow and cv2.waitKey in this function
        if not self.cv_image is None:
            depth = cv2_wrapper(self.cv_image, 64)
            depth_remove_floor = remove_depth_floor(depth)
            masked_map, mask = identify_obstacles(depth_remove_floor, depth)
            self.cartesian_points_vision, no_points = generate_point_cloud(depth, mask)
            self.cartesian_points_vision /= 100
            if no_points:
                cp = np.array([[0, -100], [0, -200]])

            else:
                self.cartesian_points_vision = np.delete(
                    self.cartesian_points_vision, 1, 1
                )
                self.cartesian_points_vision = convert_to_odom(
                    self.cartesian_points_vision, self.x, self.y, self.theta, False
                )
                cp = self.cartesian_points_vision
            # plt.imshow(
            #     masked_map,
            #     cmap=plt.get_cmap("hot"),
            #     interpolation="nearest",
            #     vmin=0,
            #     vmax=1,
            # )

            # plot vision points
            plt.scatter(cp[:, 0], cp[:, 1])

            # plot lidar points
            if len(self.cartesian_points_lidar.shape) == 1:
                cp = np.array([[0, -100], [0, -200]])
            else:
                cp = self.cartesian_points_lidar
            plt.scatter(cp[:, 0], cp[:, 1])
            plt.quiver(self.x, self.y, np.cos(self.theta), np.sin(self.theta))

            plt.xlim(-5, 5)
            plt.ylim(-5, 5)

            plt.show(block=False)
            plt.pause(0.05)
            plt.clf()

    def publish_monocular_depth_estimates(self):
        """
        Publishes cartesian points from vision
        """

        if len(self.cartesian_points_vision.shape) // 2 != 0:
            # Ensure that it is not empty

            msg = PointCloud()
            msg.header.stamp = Time()
            msg.header.frame_id = "odom"
            msg.points = [Point32(x=p[0], y=p[1]) for p in self.cartesian_points_vision]
            self.monocular_depth_pub.publish(msg)

    def publish_lidar_depth_estimates(self):
        """
        Publishes cartesian points from lidar
        """

        if len(self.cartesian_points_lidar.shape) != 1:
            # Ensure that it is not empty

            msg = PointCloud()
            msg.header.stamp = Time()
            msg.header.frame_id = "odom"
            msg.points = [Point32(x=p[0], y=p[1]) for p in self.cartesian_points_lidar]
            self.lidar_depth_pub.publish(msg)


def main(args=None):
    rclpy.init()
    n = DepthEstimation("camera/image_raw")
    rclpy.spin(n)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
