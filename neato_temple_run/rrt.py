""" An implementation of an occupancy field that you can use to implement
    your particle filter """

import rclpy
import random
import math
import time
import numpy as np
import scipy.stats as sp
from threading import Thread
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PolygonStamped, Point32, PointStamped, Point, Vector3, Pose, Quaternion
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker
from sklearn.neighbors import NearestNeighbors
from angle_helpers import quaternion_from_euler


class TreeNode(object):
    def __init__(self, pos, parent, dir):
        self.pos = pos
        self.dir = dir
        self.par = parent
        if parent is None:
            self.w = 0
        else:
            self.w = parent.w+self.find_distance(parent)

    def return_point32(self):
        return Point32(x=self.pos.x,y=self.pos.y)
    
    def find_distance(self, other):
        return (self.pos.x-other.pos.x)**2+(self.pos.y-other.pos.y)**2
    
    def __str__(self):
        return f"{self.pos.x},{self.pos.y}"

class RRT(object):
    """ Stores an occupancy field for an input map.  An occupancy field returns
        the distance to the closest obstacle for any coordinate in the map
        Attributes:
            map: the map to localize against (nav_msgs/OccupancyGrid)
            closest_occ: the distance for each entry in the OccupancyGrid to
            the closest obstacle
    """

    def __init__(self, node, occ_grid):
        self.node = node
        self.occ_grid = occ_grid
        self.resolution = self.occ_grid.map_resolution
        self.start_pos = Point32(x=0.0,y=0.0)
        self.start_dir = 0.0
        self.goal_pos = Point32(x=0.0,y=0.0)
        self.valid_goal = False
        self.path_pub = self.node.create_publisher(PolygonStamped, "path",10)
        self.goal_pub = self.node.create_publisher(PointStamped, "goal",10)
        self.tree_pub = self.node.create_publisher(PointCloud, "tree",10)
        self.goal_dir_pub = self.node.create_publisher(Marker, "goal_dir",10)
        self.curr_dir_pub = self.node.create_publisher(Marker, "curr_dir",10)

        self.tree = []
        self.tolerance = 0.2
        self.step = 0.3
        self.thresh = self.occ_grid.offset
        self.neighborhood = 0.5

        self.timer = self.node.create_timer(0.1, self.publish_path)
        self.timer3 = self.node.create_timer(0.1, self.publish_goal)
        self.timer4 = self.node.create_timer(0.1, self.publish_tree)
        self.timer5 = self.node.create_timer(0.1, self.publish_dirs)
        # self.timer2 = self.node.create_timer(2, self.rrt)
        self.path = []
        self.directions = []
        self.path_updated = False
        self.trigger = False
        self.depth = 100

        thread = Thread(target=self.loop_wrapper)
        thread.start()

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        while True:
            if self.trigger:
                self.depth = 200
                self.rrt()
                self.trigger = False
            else:
                self.depth = 1000
                self.rrt()
            time.sleep(0.1)

    def rrt(self):
        if not self.valid_goal:
            return
        self.start_pos = Point32(x=self.node.current_odom_xy_theta[0],y=self.node.current_odom_xy_theta[1])
        self.start_dir = math.atan2(math.sin(self.node.current_odom_xy_theta[2]),math.cos(self.node.current_odom_xy_theta[2]))
        closest_index = [0,math.inf]
        print(f"Current Odom: {self.node.current_odom_xy_theta}")
        print(f"Start Dir: {self.start_dir}")
        self.occ_grid.updated = True
        if self.occ_grid.updated:
            self.tree = [TreeNode(self.start_pos, None, self.start_dir)]
            counter = 0
            for i in range(self.depth):
                if closest_index[1]<self.tolerance:
                    chosen_idx = int(sp.norm.rvs(loc = 0.5*len(self.tree), scale = len(self.tree)/2))
                else:
                    chosen_idx = int(sp.norm.rvs(loc = 0.9*len(self.tree), scale = len(self.tree)/2))
                parent = self.tree[max(0,min(chosen_idx, len(self.tree)-1))]
                goal_dir_x = self.goal_pos.x-parent.pos.x
                goal_dir_y = self.goal_pos.y-parent.pos.y
                goal_dir = math.atan2(goal_dir_y, goal_dir_x)
                if self.trigger:
                    chosen_dir = sp.norm.rvs(loc = goal_dir, scale = 1)
                else:
                    chosen_dir = sp.norm.rvs(loc = (goal_dir+parent.dir)/2, scale = 1)
                dir_x = self.step*math.cos(chosen_dir)
                dir_y = self.step*math.sin(chosen_dir)
                chosen_dir = math.atan2(dir_y,dir_x)
                new_x = parent.pos.x+dir_x
                new_y = parent.pos.y+dir_y
                if self.occ_grid.get_closest_obstacle_distance(new_y,new_x)>self.thresh:
                    counter+=1
                    self.tree.append(TreeNode(Point32(x=new_x,y=new_y),parent,chosen_dir))
                    distance = math.sqrt((self.goal_pos.x-new_x)**2+(self.goal_pos.y-new_y)**2)
                    if distance<closest_index[1]:
                        closest_index=[counter,distance]
            if closest_index[1]<self.tolerance:
                self.rewire_tree()
                self.directions = []
                print(f"tree: {len(self.tree)},index: {closest_index[0]}")
                self.path = self.extract_path(self.tree[closest_index[0]])
                self.directions.reverse()
                self.path_updated = True
                self.node.path = self.path
                self.node.directions = self.directions
                # print(len(self.path))
                # break
            self.occ_grid.updated = False
            

    def extract_path(self,treenode):
        self.directions.append(treenode.dir)
        if treenode.par is None:
            return [treenode.return_point32()]
        return self.extract_path(treenode.par)+[treenode.return_point32()]
        
    def rewire_tree(self):
        for treenode in self.tree[::-1]:
            if treenode.par is not None:
                neigh, weigh, deigh = self.find_neighbors_distance(treenode)
                yeigh = [m for m,n in zip(weigh,deigh)]
                if yeigh:
                    minimum_weight = min(yeigh)
                    if minimum_weight<treenode.w:
                        minimum_index = weigh.index(minimum_weight)
                        treenode.par = neigh[minimum_index]
                        treenode.dir = math.atan2(treenode.pos.y-treenode.par.pos.y,treenode.pos.x-treenode.par.pos.x)
                        treenode.w = weigh[minimum_index]

            
    def find_neighbors_distance(self, treenode):
        neigh = []
        weigh = []
        deigh = []
        for tn in self.tree:
            dist = treenode.find_distance(tn)
            if dist<self.neighborhood and tn!=treenode:
                neigh.append(tn)
                weigh.append(dist+tn.w)
                deigh.append(abs(math.atan2(treenode.pos.y-tn.pos.y,treenode.pos.x-tn.pos.x)-tn.dir))
        return neigh, weigh, deigh

    # def dfs(self):
    #     visited = np.zeros(np.shape(self.occ_grid.closest_occ))
    #     self.path = self.dfs_helper(visited, self.start_pos,0) or [self.start_pos]
    #     self.path = [Point32(x=p.y,y=p.x) for p in self.path]
    #     # print(self.path)

    # def dfs_helper(self, visited, pos,d):
    #     x = pos.x
    #     y = pos.y
    #     [ix,iy,valid] = self.occ_grid.get_index_from_coords(x,y)
    #     # print(f"{x},{y},{ix},{iy},{valid}")
    #     visited[ix,iy] = 1
    #     if abs(self.goal_pos.x-x)<0.2 and abs(self.goal_pos.y-y)<0.2:
    #         print("found it!!!!!!!!!!!!!!!!")
    #         return [pos]
    #     elif d>400:
    #         return None
    #     path = None
    #     if path is None and ix<99 and visited[ix+1,iy]==0 and self.occ_grid.closest_occ[ix+1, iy]>0.2:
    #         path = self.dfs_helper(visited,Point32(x=x+0.05,y=y),d+1) or None
    #     if path is None and iy<99  and visited[ix,iy+1]==0 and self.occ_grid.closest_occ[ix, iy+1]>0.2:
    #         path = self.dfs_helper(visited,Point32(x=x,y=y+0.05),d+1) or None
    #     if path is None and ix>0 and visited[ix-1,iy]==0 and self.occ_grid.closest_occ[ix-1, iy]>0.2:
    #         path = self.dfs_helper(visited,Point32(x=x-0.05,y=y),d+1) or None
    #     if path is None and iy>0 and visited[ix,iy-1]==0 and self.occ_grid.closest_occ[ix, iy-1]>0.2:
    #         path = self.dfs_helper(visited,Point32(x=x,y=y-0.05),d+1) or None
    #     if path is not None:
    #         path.append(pos)
    #     return path

    def publish_tree(self):
        msg = PointCloud()
        msg.header.stamp = self.node.last_scan_timestamp or Time()
        msg.header.frame_id = "odom"
        msg.points = [p.return_point32() for p in self.tree]
        self.tree_pub.publish(msg)

    def publish_path(self):
        msg = PolygonStamped()
        msg.polygon.points = self.path
        msg.header.stamp = self.node.last_scan_timestamp or Time()
        msg.header.frame_id = "odom"
        self.path_pub.publish(msg)

    def publish_goal(self):
        msg = PointStamped()
        msg.point = Point(x=self.goal_pos.x,y=self.goal_pos.y)
        msg.header.stamp = self.node.last_scan_timestamp or Time()
        msg.header.frame_id = "odom"
        self.goal_pub.publish(msg)

    def publish_dirs(self):
        msg = Marker()
        msg.header.frame_id = 'odom'
        msg.header.stamp = self.node.last_scan_timestamp or Time()
        msg.type = Marker.ARROW
        msg.action = Marker.ADD

        # Customize the appearance
        msg.scale.x = 0.5  # Shaft diameter
        msg.scale.y = 0.1  # Head diameter
        msg.scale.z = 0.1  # Head length

        msg.pose.position = Point(x=self.start_pos.x, y=self.start_pos.y)

        # Specify the start and end points of the vector
        q = quaternion_from_euler(0,0,math.atan2(self.goal_pos.y-self.start_pos.y,self.goal_pos.x-self.start_pos.x))
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0        
        self.goal_dir_pub.publish(msg)

        # Specify the start and end points of the vector
        q = quaternion_from_euler(0,0,self.start_dir)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        msg.color.r = 0.0
        msg.color.g = 0.0
        msg.color.b = 1.0
        msg.color.a = 1.0        
        self.curr_dir_pub.publish(msg)