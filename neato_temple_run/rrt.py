""" An implementation of an occupancy field that you can use to implement
    your particle filter """

import rclpy
import random
import math
import numpy as np
import scipy.stats as sp
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PolygonStamped, Point32, PointStamped, Point
from sensor_msgs.msg import PointCloud
from sklearn.neighbors import NearestNeighbors

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
        return Point32(x=self.pos.y,y=self.pos.x)
    
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

        self.tree = []
        self.tolerance = 0.2
        self.step = 0.3
        self.thresh = self.occ_grid.offset
        self.neighborhood = 0.5

        self.timer = self.node.create_timer(0.1, self.publish_path)
        self.timer3 = self.node.create_timer(0.1, self.publish_goal)
        self.timer4 = self.node.create_timer(0.1, self.publish_tree)
        self.timer2 = self.node.create_timer(0.5, self.rrt)
        self.path = [Point32(x=15*np.random.random(),y=15*np.random.random()) for i in range(20)]
        self.path_updated = False

    def rrt(self):
        if not self.valid_goal:
            return
        self.start_pos = Point32(x=self.node.current_odom_xy_theta[1],y=self.node.current_odom_xy_theta[0])
        self.start_dir = self.node.current_odom_xy_theta[2]
        print(f"Current Odom: {self.node.current_odom_xy_theta}")
        self.occ_grid.updated = True
        if self.occ_grid.updated:
            self.tree = [TreeNode(self.start_pos, None, self.start_dir)]
            for i in range(10000):
                chosen_idx = int(sp.norm.rvs(loc = 0.9*len(self.tree), scale = len(self.tree)/2))
                parent = self.tree[max(0,min(chosen_idx, len(self.tree)-1))]
                goal_dir_x = self.goal_pos.x-parent.pos.x
                goal_dir_y = self.goal_pos.y-parent.pos.y
                goal_dir = math.atan2(goal_dir_y, goal_dir_x)
                chosen_dir = sp.norm.rvs(loc = goal_dir, scale = 1)
                chosen_dir = chosen_dir % (2*math.pi)
                dir_x = self.step*math.cos(chosen_dir)
                dir_y = self.step*math.sin(chosen_dir)
                new_x = parent.pos.x+dir_x
                new_y = parent.pos.y+dir_y
                if self.occ_grid.get_closest_obstacle_distance(new_x,new_y)>self.thresh:
                    self.tree.append(TreeNode(Point32(x=new_x,y=new_y),parent,chosen_dir))
                    if math.sqrt((self.goal_pos.x-new_x)**2+(self.goal_pos.y-new_y)**2)<self.tolerance:
                        # print(len(self.tree))
                        self.rewire_tree()
                        self.path = self.extract_path(self.tree[-1])
                        self.path_updated = True
                        # print(len(self.path))
                        break
            self.occ_grid.updated = False
            

    def extract_path(self,treenode):
        if treenode.par is None:
            return [treenode.return_point32()]
        return self.extract_path(treenode.par)+[treenode.return_point32()]
        
    def rewire_tree(self):
        for treenode in self.tree[::-1]:
            if treenode.par is not None:
                neigh, weigh = self.find_neighbors_distance(treenode)
                if weigh:
                    minimum_weight = min(weigh)
                    if minimum_weight<treenode.w:
                        minimum_index = weigh.index(minimum_weight)
                        treenode.par = neigh[minimum_index]
                        treenode.w = treenode.par.w + treenode.find_distance(treenode.par)

            
    def find_neighbors_distance(self, treenode):
        neigh = []
        weigh = []
        for tn in self.tree:
            dist = treenode.find_distance(tn)
            if dist<self.neighborhood and tn!=treenode:
                neigh.append(tn)
                weigh.append(dist+tn.w)
        return neigh, weigh

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
        msg.point = Point(x=self.goal_pos.y,y=self.goal_pos.x)
        msg.header.stamp = self.node.last_scan_timestamp or Time()
        msg.header.frame_id = "odom"
        self.goal_pub.publish(msg)