#!/usr/bin/env python3

""" An implementation of a Neato runner """

import rclpy
from threading import Thread
from rclpy.node import Node
from nav2_msgs.msg import ParticleCloud
from builtin_interfaces.msg import Time
from geometry_msgs.msg import (
    Point,
    Twist,
    Point32,
    PointStamped,
)
import math
import time
import numpy as np
from occupancy_field import OccupancyField
from helper_functions import TFHelper
from rrt import RRT
from rclpy.qos import qos_profile_sensor_data
import scipy.stats as sp
from visualization_msgs.msg import Marker
from angle_helpers import quaternion_from_euler


class NeatoRunner(Node):
    """ The class that represents a Neato Runner Node """

    def __init__(self):
        super().__init__("main")
        self.base_frame = "base_footprint"  # the frame of the robot base
        self.map_frame = "map"  # the name of the map coordinate frame
        self.odom_frame = "odom"  # the name of the odometry coordinate frame
        self.scan_topic = "scan"  # the topic where we will get laser scans from

        # Publishers for Neato and rviz
        self.particle_pub = self.create_publisher(
            ParticleCloud, "particle_cloud", qos_profile_sensor_data
        )
        self.drive_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.wp_pub = self.create_publisher(PointStamped, "next_wp", 10)
        self.wp_dir_pub = self.create_publisher(Marker, "wp_dir", 10)
        self.control_pub = self.create_publisher(Marker, "control", 10)
        self.timer = self.create_timer(0.1, self.publish_wp)
        self.timer2 = self.create_timer(0.1, self.publish_wp_dir)
        self.timer3 = self.create_timer(0.1, self.drive)
        self.timer5 = self.create_timer(0.1, self.publish_control)
        self.timer4 = self.create_timer(0.1, self.check_goal)
        self.timer6 = self.create_timer(0.1, self.updated_grid)

        self.next_wp  = Point32() # next waypoint [m]
        self.next_wp_dir = 0 # direction of next waypoint [rad]
        self.last_wp_idx = 0 # index of last waypoint
        self.ang_ctrl_out = 0 # Neato angular control output
        self.current_odom_xy_theta = [0.0, 0.0, 0.0] # current odometry pose (x,y,theta)[m,rad]
        self.current_heading = 0 # current Neato heading [rad]
        self.last_heading = 0 # last Neato heading [rad]
        self.p = 0 # proportional component
        self.i = 0 # integral component
        self.d = 0 # derivative component
        self.occupancy_field = OccupancyField(self) # occupancy field
        self.transform_helper = TFHelper(self) # transform helper
        self.number_of_goals = 2 # number of goals to chain

        # create an RRT instance (rrt), waypoint list (path), and waypoint directions list (directions) for each goal
        for i in range(self.number_of_goals):
            setattr(self, "rrt" + str(i), RRT(self, self.occupancy_field, str(i)))
            setattr(self, "path" + str(i), [Point32(x=0.0, y=0.0)])
            setattr(self, "directions" + str(i), [0])

        # we are using a thread to work around single threaded execution bottleneck
        thread = Thread(target=self.loop_wrapper)
        thread.start()

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
        We are using a separate thread to run the loop_wrapper to work around
        issues with single threaded executors in ROS2 """
        while True:
            self.run_loop()
            time.sleep(1 / 10)

    def run_loop(self):
        """ This is the main run_loop of our Neato runner. It gets new odometry poses """

        # get most recent pose
        (new_pose, _) = self.transform_helper.get_matching_odom_pose(
            self.odom_frame, self.base_frame, 0
        )
        if new_pose is None:
            # unable to get transform
            return

        # convert pose to odom list
        self.current_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(
            new_pose
        )
        # print("x: {0}, y: {1}, yaw: {2}".format(*new_odom_xy_theta))

    def shift(self, attr):
        """ This function is a circular buffer for the goal object instances """
        
        # save the left-most object
        left_most = getattr(self, attr + "0")

        # shift all other objects left by one position
        for i in range(self.number_of_goals - 1):
            setattr(self, attr + str(i), getattr(self, attr + str(i + 1)))
            if attr == "rrt":
                # re-spin RRT publishers to rematch to correct output stream
                setattr(getattr(self, attr + str(i)), "designator", str(i))
                getattr(self, attr + str(i)).create_publishers()
        
        # re-add the left-most object at the right-most location
        setattr(self, attr + str(self.number_of_goals - 1), left_most)
        if attr == "rrt":
            # re-spin RRT publishers to rematch to correct output stream
            setattr(left_most, "designator", str(self.number_of_goals - 1))
            left_most.create_publishers()

    def updated_grid(self):
        """ This function checks whether the new frontier invalidates a calculated path and reruns path planning """

        # if the occupancy grid updated
        if self.occupancy_field.updated:
            trip = False # whether a path was invalidated already

            # for each goal in order check the path
            for i in range(self.number_of_goals):
                rrt = getattr(self, "rrt" + str(i)) # get the RRT
                path = getattr(self, "path" + str(i)) # get the goal

                # check the location of the goal position
                closest = rrt.occ_grid.get_closest_obstacle_distance(
                    rrt.goal_pos.y, rrt.goal_pos.x
                )
                # if a prior path tripped, the point intersects an obstacle
                if trip or closest < rrt.thresh:
                    # reset the goal and all future goals
                    print(f"Reset {i}")
                    self.reset_goal(i)
                    trip = True
                else:
                    # check all points on the path
                    closest = math.inf
                    for p in path:
                        closest = min(
                            closest,
                            rrt.occ_grid.get_closest_obstacle_distance(p.y, p.x),
                        )
                    if closest < rrt.thresh:
                        # if goal is ok but path is not, replan the path only
                        print(f"Repathing {i}")
                        rrt.trigger_quick = True
            
            # reset updated flag
            self.occupancy_field.updated = False

    def check_goal(self):
        """ Check if the goal has been reached """
        # compute distance
        dx = self.rrt0.goal_pos.x - self.current_odom_xy_theta[0]
        dy = self.rrt0.goal_pos.y - self.current_odom_xy_theta[1]
        distance = math.sqrt(dx**2 + dy**2)
        print(f"Distance to Goal: {distance}")
        if distance < 0.5:
            # if the distance is close, shift all goals and reset the last goal
            self.shift("rrt")
            self.shift("path")
            self.shift("directions")
            self.reset_goal(self.number_of_goals - 1)

    def reset_goal(self, end):
        """ Reset the position of a goal """
        # get the RRT for the current goal
        rrt = getattr(self, "rrt" + str(end))
        if end == 0:
            # if the first goal, set the path and directions to current odometry
            path = [
                Point32(
                    x=self.current_odom_xy_theta[0], y=self.current_odom_xy_theta[1]
                )
            ]
            directions = [self.current_odom_xy_theta[2]]
        else:
            # if not the first goal, get the prior RRT, path, and directions
            rrt_last = getattr(self, "rrt" + str(end - 1))
            path = [rrt_last.goal_pos]
            directions = getattr(self, "directions" + str(end - 1))

        # compute the start orientation based on the last goal waypoint or current odometry
        rrt.start_pos = Point32(x=path[-1].x, y=path[-1].y)
        rrt.start_dir = directions[-1]

        # run a loop to randomly select a new waypoint
        counter = 0
        while True:
            # calculate a new random offset in the cardinal directions
            dx = 4 * (np.random.random() - 0.5)
            dy = 4 * (np.random.random() - 0.5)
            # find the distance and direction to the new goal
            distance = math.sqrt(dx**2 + dy**2)
            new_dir = math.atan2(dy, dx)
            # print(f"{x},{y},dir:{self.current_odom_xy_theta[2]},{new_dir}")
            # calculate the difference between the current direction and new direction
            direction_difference = abs(new_dir - directions[-1])
            # calculate the absolute goal position
            x = dx + rrt.start_pos.x
            y = dy + rrt.start_pos.y
            # create a goal point
            rrt.valid_goal = False
            rrt.goal_pos = Point32(x=x, y=y)
            # create a linear ray towards the point
            lin_x = np.linspace(rrt.start_pos.x, x, 10)
            lin_y = np.linspace(rrt.start_pos.y, y, 10)
            # loop through the ray to check for obstacles
            closest = math.inf
            for i in range(10):
                closest = min(closest,rrt.occ_grid.get_closest_obstacle_distance(lin_y[i], lin_x[i]))
            # print(f"Closest: {closest}")
            # if the goal is not out of bounds
            # if the straight line path avoids obstacles
            # if it's close enough but not too close
            # if the direction is not too deviated
            if (
                math.isfinite(closest) and (counter > 1000 or (closest > rrt.thresh
                and (direction_difference < counter*math.pi/1000)
                and 1 < distance < 5))
            ):
                # if the goal is good, set it
                rrt.goal_pos = Point32(x=x, y=y)
                rrt.valid_goal = True
                print(f"New Goal: {rrt.goal_pos}")
                if counter > 1000:
                    # if the loop is stuck, cancel
                    print("Failsafe")
                break
            counter += 1
        # trigger a new RRT pathing
        rrt.trigger_quick = True
        self.p = 0
        self.i = 0
        self.d = 0

    def drive(self):
        """ This function controls the driving of the Neato """
        # if the path updated, reset the last waypoint index
        if self.rrt0.path_updated:
            self.last_wp_idx = 0
            self.rrt0.path_updated = False
        # if a path exists
        if self.path0:
            waypoints = self.path0
            distances = []
            theta_deltas = []
            # for each waypoint, calculate the theta delta and save it
            for wp in waypoints:
                dx = wp.x - self.current_odom_xy_theta[0]
                dy = wp.y - self.current_odom_xy_theta[1]
                # print(f"fd{math.atan2(dy,dx)}")
                dt = math.atan2(dy, dx) - math.atan2(
                    math.sin(self.current_odom_xy_theta[2]),
                    math.cos(self.current_odom_xy_theta[2]),
                )
                theta_deltas.append(dt)
                dt = abs(dt)
                # print(dt)
                dis = math.sqrt(dx**2 + dy**2)
                # if the distance is far enough and the theta delta is small enough append a weight
                if dis > 0.5 and dt < math.pi * 0.75:
                    distances.append(dis + dt / 3)
                else:
                    distances.append(math.inf)

            # calculate the index of lowest weight within the remaining points
            index = distances.index(min(distances[self.last_wp_idx :]))
            # if there is no best waypoint, choose one two indices ahead
            if math.isinf(distances[index]):
                index = min(self.last_wp_idx + 2, len(distances) - 1)
            else:
                self.last_wp_idx = index
            
            # extract the new waypoint and waypoint direction
            self.next_wp  = waypoints[index]
            self.next_wp_dir = self.directions0[index]

            # calculate the angle deviation with wrapping
            angle_deviation = theta_deltas[index]
            if angle_deviation<-math.pi:
                angle_deviation+=2*math.pi
            elif angle_deviation>math.pi:
                angle_deviation-=2*math.pi

            # calculate current heading
            self.current_heading = math.atan2(
                math.sin(self.current_odom_xy_theta[2]),
                math.cos(self.current_odom_xy_theta[2]),
            )

            # calculate PID
            self.p = angle_deviation
            kp = 1
            self.i += angle_deviation
            ki = 0.0
            self.d = math.atan2(
                math.sin(self.current_heading - self.last_heading),
                math.cos(self.current_heading - self.last_heading),
            )
            kd = 0.0

            # save prior heading
            self.last_heading = self.current_heading

            # calculate control output
            self.ang_ctrl_out = kp * self.p + ki * self.i + kd * self.d
            print(self.ang_ctrl_out)

            # publish Twist command
            cmd_vel = Twist()
            cmd_vel.linear.x = float(max(0, max(0.1, 1 - abs(self.ang_ctrl_out))))
            cmd_vel.angular.z = float(max(-2, min(2, self.ang_ctrl_out)))
            self.drive_pub.publish(cmd_vel)
            # print("Publish Drive")

    def publish_wp(self):
        """ Publish the next waypoint as a point """
        msg = PointStamped()
        msg.point = Point(x=self.next_wp.x, y=self.next_wp.y)
        msg.header.stamp = Time()
        msg.header.frame_id = "odom"
        self.wp_pub.publish(msg)

    def publish_wp_dir(self):
        """ Publish the next waypoint direction as an arrow marker """
        msg = Marker()
        msg.header.frame_id = "odom"
        msg.header.stamp = Time()
        msg.type = Marker.ARROW
        msg.action = Marker.ADD

        # Customize the appearance
        msg.scale.x = 0.5  # Shaft diameter
        msg.scale.y = 0.1  # Head diameter
        msg.scale.z = 0.1  # Head length

        msg.pose.position = Point(x=self.next_wp.x, y=self.next_wp.y)

        # Specify the start and end points of the vector
        q = quaternion_from_euler(0, 0, self.next_wp_dir)
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
        """ Publish the control output as an arrow marker """
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
            0, 0, self.current_odom_xy_theta[2] + 1.5 * self.ang_ctrl_out
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
    # Spin the NeatoRunner node
    rclpy.init()
    n = NeatoRunner()
    rclpy.spin(n)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
