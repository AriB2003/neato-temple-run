#!/usr/bin/env python3

""" This is the starter code for the robot localization project """

import rclpy
from threading import Thread
# from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import ParticleCloud, Particle
from nav2_msgs.msg import Particle as Nav2Particle
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion, Twist, Point32, PointStamped
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
    """ The class that represents a Particle Filter ROS Node
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
        super().__init__('main')
        self.base_frame = "base_footprint"   # the frame of the robot base
        self.map_frame = "map"          # the name of the map coordinate frame
        self.odom_frame = "odom"        # the name of the odometry coordinate frame
        self.scan_topic = "scan"        # the topic where we will get laser scans from 


        self.particle_cloud = []
        self.wp = Point32()
        self.chosen_dir = 0
        # publish the current particle cloud.  This enables viewing particles in rviz.
        self.particle_pub = self.create_publisher(ParticleCloud, "particle_cloud", qos_profile_sensor_data)
        self.drive_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.wp_pub = self.create_publisher(PointStamped, "next_wp", 10)
        self.wp_dir_pub = self.create_publisher(Marker, "wp_dir", 10)
        self.timer = self.create_timer(0.1, self.publish_wp)
        self.timer2 = self.create_timer(0.1, self.publish_wp_dir)

        # laser_subscriber listens for data from the lidar
        self.create_subscription(LaserScan, self.scan_topic, self.scan_received, 10)

        # this is used to keep track of the timestamps coming from bag files
        # knowing this information helps us set the timestamp of our map -> odom
        # transform correctly
        self.last_scan_timestamp = None
        # this is the current scan that our run_loop should process
        self.scan_to_process = None
        # your particle cloud will go here
        

        self.current_odom_xy_theta = [0.0,0.0,0.0]
        self.occupancy_field = OccupancyField(self)
        self.transform_helper = TFHelper(self)
        self.rrt = RRT(self, self.occupancy_field)
        self.last_index = 0

        # we are using a thread to work around single threaded execution bottleneck
        thread = Thread(target=self.loop_wrapper)
        thread.start()
        # self.transform_update_timer = self.create_timer(0.05, self.pub_latest_transform)

        # PARAMETERS
        self.n_particles = 300          # the number of particles to use
        self.d_thresh = 0.2             # the amount of linear movement before performing an update
        self.a_thresh = math.pi/6       # the amount of angular movement before performing an update

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        while True:
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):
        """ This is the main run_loop of our particle filter.  It checks to see if
            any scans are ready and to be processed and will call several helper
            functions to complete the processing.
            
            You do not need to modify this function, but it is helpful to understand it.
        """
        if self.scan_to_process is None:
            return
        msg = self.scan_to_process
        
        (new_pose, delta_t) = self.transform_helper.get_matching_odom_pose(self.odom_frame,
                                                                           self.base_frame,
                                                                           msg.header.stamp)
        if new_pose is None:
            # we were unable to get the pose of the robot corresponding to the scan timestamp
            if delta_t is not None and delta_t < Duration(seconds=0.0):
                # we will never get this transform, since it is before our oldest one
                self.scan_to_process = None
            return
        
        (r, theta) = self.transform_helper.convert_scan_to_polar_in_robot_frame(msg, self.base_frame)
        # print("r[0]={0}, theta[0]={1}".format(r[0], theta[0]))
        # clear the current scan so that we can process the next one
        self.scan_to_process = None

        self.odom_pose = new_pose
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        # print("x: {0}, y: {1}, yaw: {2}".format(*new_odom_xy_theta))

        self.current_odom_xy_theta = new_odom_xy_theta

        self.drive()
        self.reset_goal()
            
    def reset_goal(self):
        dx = self.rrt.goal_pos.x-self.current_odom_xy_theta[1]
        dy = self.rrt.goal_pos.y-self.current_odom_xy_theta[0]
        distance = math.sqrt(dx**2+dy**2)
        print(f"Distance to Goal: {distance}")
        if distance<0.5:
            counter = 0
            while True:
                x=self.occupancy_field.map_width*np.random.random()*self.occupancy_field.map_resolution+self.occupancy_field.map_origin_x
                y=self.occupancy_field.map_height*np.random.random()*self.occupancy_field.map_resolution+self.occupancy_field.map_origin_y
                dx = x-self.current_odom_xy_theta[1]
                dy = y-self.current_odom_xy_theta[0]
                distance = math.sqrt(dx**2+dy**2)
                new_dir = math.atan2(x-self.current_odom_xy_theta[0],y-self.current_odom_xy_theta[1])
                # print(f"{x},{y},dir:{self.current_odom_xy_theta[2]},{new_dir}")
                direction_difference = abs(new_dir-math.atan2(math.sin(self.current_odom_xy_theta[2]),math.cos(self.current_odom_xy_theta[2])))
                self.rrt.valid_goal = False
                self.rrt.goal_pos = Point32(x=x,y=y)
                if self.rrt.occ_grid.get_closest_obstacle_distance(x,y)>1.5*self.rrt.thresh and (direction_difference<0.75 or counter>100) and 1<distance<3:
                    self.rrt.goal_pos = Point32(x=x,y=y)
                    self.rrt.valid_goal = True
                    print(f"New Goal: {self.rrt.goal_pos}")
                    if counter>100:
                        print("Failsafe")
                    break
                counter+=1

    def drive(self):
        if self.rrt.path_updated:
            self.last_index=0
            self.rrt.path_updated = False
        if self.rrt.path:
            waypoints = self.rrt.path
            distances = []
            dts = []
            for wp in waypoints:
                dx = wp.x-self.current_odom_xy_theta[0]
                dy = wp.y-self.current_odom_xy_theta[1]
                dt = math.atan2(math.sin(self.current_odom_xy_theta[2]),math.cos(self.current_odom_xy_theta[2]))-math.atan2(dy,dx)
                distances.append(math.sqrt(dx**2+dy**2)+abs(dt))
                dts.append(dt)
            index = distances.index(min(distances[self.last_index:]))
            self.wp = waypoints[index]
            self.chosen_dir = self.rrt.directions[index]
            direction = dts[index]
            self.last_index = index
            cmd_vel = Twist()
            cmd_vel.linear.x = float(0.4)
            cmd_vel.angular.z = float(-direction)
            self.drive_pub.publish(cmd_vel)
            print("Publish Drive")


    def scan_received(self, msg):
        self.last_scan_timestamp = msg.header.stamp
        # we throw away scans until we are done processing the previous scan
        # self.scan_to_process is set to None in the run_loop 
        if self.scan_to_process is None:
            self.scan_to_process = msg

    def publish_particles(self, timestamp):
        msg = ParticleCloud()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = timestamp
        for p in self.particle_cloud:
            msg.particles.append(Nav2Particle(pose=p.as_pose(), weight=p.w))
        self.particle_pub.publish(msg)

    def publish_wp(self):
        msg = PointStamped()
        msg.point = Point(x=self.wp.x, y=self.wp.y)
        msg.header.stamp = self.last_scan_timestamp or Time()
        msg.header.frame_id = "odom"
        self.wp_pub.publish(msg)

    def publish_wp_dir(self):
        msg = Marker()
        msg.header.frame_id = 'odom'
        msg.header.stamp = self.last_scan_timestamp or Time()
        msg.type = Marker.ARROW
        msg.action = Marker.ADD

        # Customize the appearance
        msg.scale.x = 0.5  # Shaft diameter
        msg.scale.y = 0.1  # Head diameter
        msg.scale.z = 0.1  # Head length

        msg.pose.position = Point(x=self.wp.x, y=self.wp.y)

        # Specify the start and end points of the vector
        q = quaternion_from_euler(0,0,self.chosen_dir)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        msg.color.r = 1.0
        msg.color.g = 1.0
        msg.color.b = 0.0
        msg.color.a = 1.0        
        self.wp_dir_pub.publish(msg)

def main(args=None):
    rclpy.init()
    n = ParticleFilter()
    rclpy.spin(n)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
