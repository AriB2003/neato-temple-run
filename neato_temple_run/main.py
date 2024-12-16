#!/usr/bin/env python3

""" This is the starter code for the robot localization project """

import rclpy
import copy
from threading import Thread

# from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import ParticleCloud, Particle
from nav2_msgs.msg import Particle as Nav2Particle
from builtin_interfaces.msg import Time
from geometry_msgs.msg import (
    PoseWithCovarianceStamped,
    Pose,
    Point,
    Quaternion,
    Twist,
    Point32,
    PointStamped,
)
from rclpy.duration import Duration
import math
import time
import random
import numpy as np
from occupancy_field import OccupancyField
from helper_functions import TFHelper
from rrt import RRT
from rclpy.qos import qos_profile_sensor_data
import scipy.stats as sp
from visualization_msgs.msg import Marker
from angle_helpers import quaternion_from_euler


class ParticleFilter(Node):
    """The class that represents a Particle Filter ROS Node
    Attributes list:
        base_frame: the name of the robot base coordinate frame (should be "base_footprint" for most robots)
        map_frame: the name of the map coordinate frame (should be "map" in most cases)
        odom_frame: the name of the odometry coordinate frame (should be "odom" in most cases)
        scan_topic: the name of the scan topic to listen to (should be "scan" in most cases)
        n_particles: the number of particles in the filter
        d_thresh: the amount of linear movement before triggering a filter update
        a_thresh: the amount of angular movement before triggering a filter update
        pose_listener: a subscriber that listens for new approximate pose estimates (i.e. generated through the rviz GUI)
        particle_pub: a publisher for the particle cloud
        last_scan_timestamp: this is used to keep track of the clock when using bags
        scan_to_process: the scan that our run_loop should process next
        occupancy_field: this helper class allows you to query the map for distance to closest obstacle
        transform_helper: this helps with various transform operations (abstracting away the tf2 module)
        particle_cloud: a list of particles representing a probability distribution over robot poses
        current_odom_xy_theta: the pose of the robot in the odometry frame when the last filter update was performed.
                               The pose is expressed as a list [x,y,theta] (where theta is the yaw)
        thread: this thread runs your main loop
    """

    def __init__(self):
        super().__init__("main")
        self.base_frame = "base_footprint"  # the frame of the robot base
        self.map_frame = "map"  # the name of the map coordinate frame
        self.odom_frame = "odom"  # the name of the odometry coordinate frame
        self.scan_topic = "scan"  # the topic where we will get laser scans from

        self.wp = Point32()
        self.chosen_dir = 0
        # publish the current particle cloud.  This enables viewing particles in rviz.
        self.particle_pub = self.create_publisher(
            ParticleCloud, "particle_cloud", qos_profile_sensor_data
        )
        self.drive_pub = self.create_publisher(Twist, "cmd_vel", 40)
        self.wp_pub = self.create_publisher(PointStamped, "next_wp", 10)
        self.wp_dir_pub = self.create_publisher(Marker, "wp_dir", 10)
        self.control_pub = self.create_publisher(Marker, "control", 40)
        self.timer = self.create_timer(0.1, self.publish_wp)
        self.timer2 = self.create_timer(0.1, self.publish_wp_dir)
        self.timer3 = self.create_timer(1 / 40, self.drive)
        self.timer5 = self.create_timer(1 / 40, self.publish_control)
        self.timer4 = self.create_timer(0.1, self.check_goal)
        self.timer6 = self.create_timer(0.1, self.updated_grid)

        self.current_odom_xy_theta = [0.0, 0.0, 0.0]
        self.occupancy_field = OccupancyField(self)
        self.transform_helper = TFHelper(self)
        self.number = 2
        for i in range(self.number):
            setattr(self, "rrt" + str(i), RRT(self, self.occupancy_field, str(i)))
            setattr(self, "path" + str(i), [Point32(x=0.0, y=0.0)])
            setattr(self, "directions" + str(i), [0])
        self.last_index = 0
        self.control_out = 0

        # while self.occupancy_field.updated == False:
        #     time.sleep(0.1)
        # we are using a thread to work around single threaded execution bottleneck
        thread = Thread(target=self.loop_wrapper)
        thread.start()
        # self.transform_update_timer = self.create_timer(0.05, self.pub_latest_transform)

        # PARAMETERS
        self.n_particles = 300  # the number of particles to use
        self.d_thresh = 0.2  # the amount of linear movement before performing an update
        self.a_thresh = (
            math.pi / 6
        )  # the amount of angular movement before performing an update

        self.heading = 0
        self.last_heading = 0
        self.p = 0
        self.i = 0
        self.d = 0

    def loop_wrapper(self):
        """This function takes care of calling the run_loop function repeatedly.
        We are using a separate thread to run the loop_wrapper to work around
        issues with single threaded executors in ROS2"""
        while True:
            self.run_loop()
            time.sleep(1 / 40)

    def run_loop(self):
        """This is the main run_loop of our particle filter.  It checks to see if
        any scans are ready and to be processed and will call several helper
        functions to complete the processing.

        You do not need to modify this function, but it is helpful to understand it.
        """

        (new_pose, delta_t) = self.transform_helper.get_matching_odom_pose(
            self.odom_frame, self.base_frame, 0
        )
        if new_pose is None:
            # we were unable to get the pose of the robot corresponding to the scan timestamp
            return

        self.odom_pose = new_pose
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(
            self.odom_pose
        )
        # print("x: {0}, y: {1}, yaw: {2}".format(*new_odom_xy_theta))

        self.current_odom_xy_theta = new_odom_xy_theta

    def shift(self, attr):
        left_most = getattr(self, attr + "0")
        for i in range(self.number - 1):
            setattr(self, attr + str(i), getattr(self, attr + str(i + 1)))
            if attr == "rrt":
                setattr(getattr(self, attr + str(i)), "designator", str(i))
                getattr(self, attr + str(i)).create_publishers()
        setattr(self, attr + str(self.number - 1), left_most)
        if attr == "rrt":
            setattr(left_most, "designator", str(self.number - 1))
            left_most.create_publishers()

    def updated_grid(self):
        if self.occupancy_field.updated:
            trip = False
            for i in range(self.number):
                rrt = getattr(self, "rrt" + str(i))
                path = getattr(self, "path" + str(i))
                closest = rrt.occ_grid.get_closest_obstacle_distance(
                    rrt.goal_pos.y, rrt.goal_pos.x
                )
                if trip or not math.isnan(closest) and closest < rrt.thresh:
                    print(f"Reset {i}")
                    self.reset_goal(i)
                    trip = True
                else:
                    closest = math.inf
                    for p in path:
                        closest = min(
                            closest,
                            rrt.occ_grid.get_closest_obstacle_distance(p.y, p.x),
                        )
                    if not math.isnan(closest) and closest < rrt.thresh:
                        print(f"Repathing {i}")
                        rrt.trigger_quick = True
            self.occupancy_field.updated = False

    def check_goal(self):
        dx = self.rrt0.goal_pos.x - self.current_odom_xy_theta[0]
        dy = self.rrt0.goal_pos.y - self.current_odom_xy_theta[1]
        distance = math.sqrt(dx**2 + dy**2)
        print(f"Distance to Goal: {distance}")
        if distance < 0.5:
            self.shift("rrt")
            self.shift("path")
            self.shift("directions")
            self.reset_goal(self.number - 1)

    def reset_goal(self, end):
        counter = 0
        rrt = getattr(self, "rrt" + str(end))
        if end == 0:
            path = [
                Point32(
                    x=self.current_odom_xy_theta[0], y=self.current_odom_xy_theta[1]
                )
            ]
            directions = [self.current_odom_xy_theta[2]]
        else:
            rrt_last = getattr(self, "rrt" + str(end - 1))
            path = [rrt_last.goal_pos]
            directions = getattr(self, "directions" + str(end - 1))
        rrt.start_pos = Point32(x=path[-1].x, y=path[-1].y)
        rrt.start_dir = directions[-1]
        while True:
            dx = 4 * (np.random.random() - 0.5)
            dy = 4 * (np.random.random() - 0.5)
            distance = math.sqrt(dx**2 + dy**2)
            new_dir = math.atan2(dy, dx)
            # print(f"{x},{y},dir:{self.current_odom_xy_theta[2]},{new_dir}")
            direction_difference = abs(new_dir - directions[-1])
            x = dx + rrt.start_pos.x
            y = dy + rrt.start_pos.y
            rrt.valid_goal = False
            rrt.goal_pos = Point32(x=x, y=y)
            lin_x = np.linspace(rrt.start_pos.x, x, 10)
            lin_y = np.linspace(rrt.start_pos.y, y, 10)
            closest = math.inf
            for i in range(10):
                closest = min(
                    closest,
                    rrt.occ_grid.get_closest_obstacle_distance(lin_y[i], lin_x[i]),
                )
            if (
                not math.isnan(closest)
                and closest > rrt.thresh
                and (direction_difference < 1 or counter > 1000)
                and 1 < distance < 5
            ):
                rrt.goal_pos = Point32(x=x, y=y)
                rrt.valid_goal = True
                print(f"New Goal: {rrt.goal_pos}")
                if counter > 100:
                    print("Failsafe")
                break
            counter += 1
        rrt.trigger_quick = True
        self.p = 0
        self.i = 0
        self.d = 0

    def drive(self):
        if self.rrt0.path_updated:
            self.last_index = 0
            self.rrt0.path_updated = False
        if self.path0:
            waypoints = self.path0
            distances = []
            dts = []
            for wp in waypoints:
                dx = wp.x - self.current_odom_xy_theta[0]
                dy = wp.y - self.current_odom_xy_theta[1]
                # print(f"fd{math.atan2(dy,dx)}")
                dt = math.atan2(dy, dx) - math.atan2(
                    math.sin(self.current_odom_xy_theta[2]),
                    math.cos(self.current_odom_xy_theta[2]),
                )
                dts.append(dt)
                dt = abs(dt)
                # print(dt)
                dis = math.sqrt(dx**2 + dy**2)
                if dis > 0.5 and dt < math.pi * 0.75:
                    distances.append(dis + dt / 3)
                else:
                    distances.append(math.inf)

            index = distances.index(min(distances[self.last_index :]))
            if math.isinf(distances[index]):
                index = min(self.last_index + 2, len(distances) - 1)
            else:
                self.last_index = index
            self.wp = waypoints[index]
            self.chosen_dir = self.directions0[index]
            direction = dts[index]

            self.heading = math.atan2(
                math.sin(self.current_odom_xy_theta[2]),
                math.cos(self.current_odom_xy_theta[2]),
            )
            self.p = direction
            kp = 1
            self.i += direction
            ki = 0.0
            self.d = math.atan2(
                math.sin(self.heading - self.last_heading),
                math.cos(self.heading - self.last_heading),
            )
            kd = 0.0
            self.last_heading = self.heading

            self.control_out = kp * self.p + ki * self.i + kd * self.d
            print(self.control_out)

            cmd_vel = Twist()
            cmd_vel.linear.x = float(max(0, max(0.1, 0.5 - abs(self.control_out))))
            cmd_vel.angular.z = float(max(-0.5, min(0.5, self.control_out)))
            self.drive_pub.publish(cmd_vel)
            # print("Publish Drive")

    def publish_wp(self):
        msg = PointStamped()
        msg.point = Point(x=self.wp.x, y=self.wp.y)
        msg.header.stamp = Time()
        msg.header.frame_id = "odom"
        self.wp_pub.publish(msg)

    def publish_wp_dir(self):
        msg = Marker()
        msg.header.frame_id = "odom"
        msg.header.stamp = Time()
        msg.type = Marker.ARROW
        msg.action = Marker.ADD

        # Customize the appearance
        msg.scale.x = 0.5  # Shaft diameter
        msg.scale.y = 0.1  # Head diameter
        msg.scale.z = 0.1  # Head length

        msg.pose.position = Point(x=self.wp.x, y=self.wp.y)

        # Specify the start and end points of the vector
        q = quaternion_from_euler(0, 0, self.chosen_dir)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        msg.color.r = 1.0
        msg.color.g = 1.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        self.wp_dir_pub.publish(msg)

    def publish_control(self):
        msg = Marker()
        msg.header.frame_id = "odom"
        msg.header.stamp = Time()
        msg.type = Marker.ARROW
        msg.action = Marker.ADD

        # Customize the appearance
        msg.scale.x = 0.5  # Shaft diameter
        msg.scale.y = 0.1  # Head diameter
        msg.scale.z = 0.1  # Head length

        msg.pose.position = Point(
            x=self.current_odom_xy_theta[0], y=self.current_odom_xy_theta[1]
        )

        # Specify the start and end points of the vector
        q = quaternion_from_euler(
            0, 0, self.current_odom_xy_theta[2] + 1.5 * self.control_out
        )
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        msg.color.r = 0.0
        msg.color.g = 1.0
        msg.color.b = 1.0
        msg.color.a = 1.0
        self.wp_dir_pub.publish(msg)


def main(args=None):
    rclpy.init()
    n = ParticleFilter()
    rclpy.spin(n)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
