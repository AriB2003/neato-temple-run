""" An implementation of an occupancy field that you can use to implement
    your particle filter """

import rclpy
import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PolygonStamped, Point32
from sklearn.neighbors import NearestNeighbors

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
        self.goal_pos = Point32(x=4.5,y=4.5)
        self.path_pub = self.node.create_publisher(PolygonStamped, "path",10)
        self.timer = self.node.create_timer(0.1, self.publish_path)
        self.timer2 = self.node.create_timer(0.5, self.dfs)
        self.path = [Point32(x=15*np.random.random(),y=15*np.random.random()) for i in range(20)]

    def dfs(self):
        visited = np.zeros(np.shape(self.occ_grid.closest_occ))
        self.path = self.dfs_helper(visited, self.start_pos,0) or [self.start_pos]
        self.path = [Point32(x=p.y,y=p.x) for p in self.path]
        # print(self.path)

    def dfs_helper(self, visited, pos,d):
        x = pos.x
        y = pos.y
        [ix,iy,valid] = self.occ_grid.get_index_from_coords(x,y)
        # print(f"{x},{y},{ix},{iy},{valid}")
        visited[ix,iy] = 1
        if abs(self.goal_pos.x-x)<0.2 and abs(self.goal_pos.y-y)<0.2:
            print("found it!!!!!!!!!!!!!!!!")
            return [pos]
        elif d>400:
            return None
        path = None
        if path is None and ix<99 and visited[ix+1,iy]==0 and self.occ_grid.closest_occ[ix+1, iy]>0.2:
            path = self.dfs_helper(visited,Point32(x=x+0.05,y=y),d+1) or None
        if path is None and iy<99  and visited[ix,iy+1]==0 and self.occ_grid.closest_occ[ix, iy+1]>0.2:
            path = self.dfs_helper(visited,Point32(x=x,y=y+0.05),d+1) or None
        if path is None and ix>0 and visited[ix-1,iy]==0 and self.occ_grid.closest_occ[ix-1, iy]>0.2:
            path = self.dfs_helper(visited,Point32(x=x-0.05,y=y),d+1) or None
        if path is None and iy>0 and visited[ix,iy-1]==0 and self.occ_grid.closest_occ[ix, iy-1]>0.2:
            path = self.dfs_helper(visited,Point32(x=x,y=y-0.05),d+1) or None
        if path is not None:
            path.append(pos)
        return path

    def publish_path(self):
        msg = PolygonStamped()
        msg.polygon.points = self.path
        msg.header.stamp = self.node.last_scan_timestamp or Time()
        msg.header.frame_id = "odom"
        self.path_pub.publish(msg)