""" An implementation of an RRT path planner """

import math
import time
import scipy.stats as sp
from threading import Thread
from builtin_interfaces.msg import Time
from geometry_msgs.msg import (
    PolygonStamped,
    Point32,
    PointStamped,
    Point,
)
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker
from angle_helpers import quaternion_from_euler


class TreeNode(object):
    """ Class for point nodes in the tree """
    def __init__(self, pos, parent, dir):
        self.pos = pos # position
        self.dir = dir # direction
        self.par = parent # parent TreeNode
        if parent is None: # weight
            self.w = 0
        else:
            self.w = parent.w + self.find_distance(parent)

    def return_point32(self):
        """ Return TreeNode as Point32 """
        return Point32(x=self.pos.x, y=self.pos.y)

    def find_distance(self, other):
        """ Find distance to other TreeNode """
        return (self.pos.x - other.pos.x) ** 2 + (self.pos.y - other.pos.y) ** 2

    def __str__(self):
        """ String representation """
        return f"{self.pos.x},{self.pos.y}"


class RRT(object):
    """ Stores an occupancy field for an input map. An occupancy field returns
    the distance to the closest obstacle for any coordinate in the map """

    def __init__(self, node, occ_grid, designator):
        self.node = node # NeatoRunner node
        self.designator = designator # designator for tracking goal number
        self.occ_grid = occ_grid # OccupancyGrid
        self.resolution = self.occ_grid.map_resolution # resolution of the map
        self.start_pos = Point32(x=0.0, y=0.0) # start position
        self.start_dir = 0.0 # start direction
        self.goal_pos = Point32(x=0.0, y=0.0) # goal position
        self.valid_goal = False # goal valid

        self.create_publishers() # create publishers
        self.timer = self.node.create_timer(0.1, self.publish_path)
        self.timer3 = self.node.create_timer(0.1, self.publish_goal)
        self.timer4 = self.node.create_timer(0.1, self.publish_tree)
        self.timer5 = self.node.create_timer(0.1, self.publish_dirs)

        self.tree = [] # RRT tree
        self.tolerance = 0.2 # goal reached tolerance
        self.step = 0.3 # step size
        self.thresh = self.occ_grid.offset * 1.5 # obstacle threshold
        self.neighborhood = 0.5 # neighborhood radius
        self.depth = 100 # maximum depth flag

        self.path = [] # found path
        self.directions = [] # waypoint directions
        self.path_updated = False # path updated flag
        self.trigger_quick = False # quick recalculation trigger
        self.trigger_long = False # slow recalculation trigger
        self.first = False # first run flag
        self.success = False # success flag

        # multithreading parallelization
        thread = Thread(target=self.loop_wrapper)
        thread.start()

    def create_publishers(self):
        """ Recreate the publishers with a new designator """
        self.path_pub = self.node.create_publisher(
            PolygonStamped, "path" + self.designator, 10
        )
        self.goal_pub = self.node.create_publisher(
            PointStamped, "goal" + self.designator, 10
        )
        self.tree_pub = self.node.create_publisher(
            PointCloud, "tree" + self.designator, 10
        )
        self.goal_dir_pub = self.node.create_publisher(
            Marker, "goal_dir" + self.designator, 10
        )
        self.curr_dir_pub = self.node.create_publisher(
            Marker, "curr_dir" + self.designator, 10
        )

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
        We are using a separate thread to run the loop_wrapper to work around
        issues with single threaded executors in ROS2 """
        while True:
            if self.trigger_quick:
                # if the quick pathing triggered, run a fast RRT, then trigger long
                self.depth = 2000
                self.step = 0.5
                self.trigger_quick = False
                self.first = True
                self.rrt()
                self.first = False
                self.trigger_long = True
            if self.trigger_long:
                # if the long pathing triggered, run a long RRT
                self.trigger_long = False
                self.depth = 5000
                self.step = 0.3
                self.success = False
                counter = 0
                # rerun until next trigger or 50 iterations
                while not self.success and not self.trigger_quick and counter < 50:
                    self.rrt()
                    time.sleep(0.1)
                    counter += 1
            time.sleep(0.01)

    def rrt(self):
        """ Run the RRT algorithm """
        if not self.valid_goal:
            # return if goal invalid
            return
        print(f"Current Odom: {self.node.current_odom_xy_theta}")
        print(f"Start Dir: {self.start_dir}")
        # initialize the root node
        self.tree = [TreeNode(self.start_pos, None, self.start_dir)]
        # track the closest node to goal
        closest_index = [0, math.inf]
        counter = 0
        for _ in range(self.depth):
            if self.trigger_quick:
                # cancel if retriggered
                return
            # if close to the goal, use smaller steps 
            if closest_index[1] < self.tolerance:
                chosen_idx = int(
                    sp.norm.rvs(loc=0.5 * len(self.tree), scale=len(self.tree) / 2)
                )
            else:
                chosen_idx = int(
                    sp.norm.rvs(loc=0.9 * len(self.tree), scale=len(self.tree) / 2)
                )
            # get random parent node using distribution
            parent = self.tree[max(0, min(chosen_idx, len(self.tree) - 1))]
            # calculate direction from parent to goal
            goal_dir_x = self.goal_pos.x - parent.pos.x
            goal_dir_y = self.goal_pos.y - parent.pos.y
            goal_dir = math.atan2(goal_dir_y, goal_dir_x)
            # if in the quick RRT, aim straight for the goal, else aim for smoothness
            if self.first:
                chosen_dir = sp.norm.rvs(loc=goal_dir, scale=1)
            else:
                chosen_dir = sp.norm.rvs(loc=(goal_dir + parent.dir) / 2, scale=1)
            # create a new node one step in the direction chosen
            dir_x = self.step * math.cos(chosen_dir)
            dir_y = self.step * math.sin(chosen_dir)
            chosen_dir = math.atan2(dir_y, dir_x)
            new_x = parent.pos.x + dir_x
            new_y = parent.pos.y + dir_y
            # check if node is valid
            if self.occ_grid.get_closest_obstacle_distance(new_y, new_x) > self.thresh:
                counter += 1
                # add to tree
                self.tree.append(
                    TreeNode(Point32(x=new_x, y=new_y), parent, chosen_dir)
                )
                distance = math.sqrt(
                    (self.goal_pos.x - new_x) ** 2 + (self.goal_pos.y - new_y) ** 2
                )
                # if the distance if closer to the goal, note it down
                if distance < closest_index[1]:
                    closest_index = [counter, distance]
                    # if close enough to goal, break the loop
                    if distance < self.tolerance:
                        break
        # if the RRT succeeded
        if closest_index[1] < self.tolerance:
            # rewire the tree
            self.rewire_tree()
            if self.trigger_quick:
                # cancel if triggered
                return
            self.directions = []
            print(f"tree: {len(self.tree)},index: {closest_index[0]}")
            # extract the path and directions
            self.path = self.extract_path(self.tree[closest_index[0]])
            self.directions.reverse()
            # set the path and directions within the NeatoRunner class, flag updates
            setattr(self.node, "path" + self.designator, self.path)
            self.path_updated = True
            setattr(self.node, "directions" + self.designator, self.directions)
            self.success = True

    def extract_path(self, treenode):
        """ Extract the path from the tree recursively """
        self.directions.append(treenode.dir) # save waypoint directions
        if treenode.par is None:
            return [treenode.return_point32()] # return a list of Point32
        return self.extract_path(treenode.par) + [treenode.return_point32()]

    def rewire_tree(self):
        """ Rewire the tree for traversal speed """
        for treenode in self.tree[::-1]:
            # traverse in reverse from goal to start
            if self.trigger_quick:
                # cancel if triggered
                return
            if treenode.par is not None:
                # find neighbors, weights, and theta difference
                neighbors, weights, angles = self.find_neighbors_distance(treenode)
                preferences = [m for m, n in zip(weights, angles)] # can be used to apply a weighting to the two factors
                if preferences:
                    # if nodes exist in the neighborhood, find the minimum weight
                    minimum_weight = min(preferences)
                    if minimum_weight < treenode.w:
                        # reset the node's properties with the new parent
                        minimum_index = weights.index(minimum_weight)
                        treenode.par = neighbors[minimum_index]
                        treenode.dir = math.atan2(
                            treenode.pos.y - treenode.par.pos.y,
                            treenode.pos.x - treenode.par.pos.x,
                        )
                        treenode.w = weights[minimum_index]

    def find_neighbors_distance(self, treenode):
        """ Find the neighbors, weights, and angles, in a neighborhood """
        neighbors = []
        weights = []
        angles = []
        for tn in self.tree:
            # loop through all nodes and find within neighborhood
            dist = treenode.find_distance(tn)
            if dist < self.neighborhood and tn != treenode:
                neighbors.append(tn)
                weights.append(dist + tn.w)
                angles.append(
                    abs(
                        math.atan2(treenode.pos.y - tn.pos.y, treenode.pos.x - tn.pos.x)
                        - tn.dir
                    )
                )
        return neighbors, weights, angles

    def publish_tree(self):
        """ Publish the tree as a PointCloud """
        msg = PointCloud()
        msg.header.stamp = Time()
        msg.header.frame_id = "odom"
        msg.points = [p.return_point32() for p in self.tree]
        self.tree_pub.publish(msg)

    def publish_path(self):
        """ Publish the path as a PolygonStamped """
        msg = PolygonStamped()
        msg.polygon.points = self.path
        msg.header.stamp = Time()
        msg.header.frame_id = "odom"
        self.path_pub.publish(msg)

    def publish_goal(self):
        """ Publish the goal as a PointStamped """
        msg = PointStamped()
        msg.point = Point(x=self.goal_pos.x, y=self.goal_pos.y)
        msg.header.stamp = Time()
        msg.header.frame_id = "odom"
        self.goal_pub.publish(msg)

    def publish_dirs(self):
        """ Publish the neato orientation and waypoint direction as arrow Markers """
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
            x=self.node.current_odom_xy_theta[0], y=self.node.current_odom_xy_theta[1]
        )

        # Specify the start and end points of the vector
        q = quaternion_from_euler(
            0,
            0,
            math.atan2(
                self.node.next_wp.y - self.node.current_odom_xy_theta[1],
                self.node.next_wp.x - self.node.current_odom_xy_theta[0],
            ),
        )
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
        q = quaternion_from_euler(
            0,
            0,
            math.atan2(
                math.sin(self.node.current_odom_xy_theta[2]),
                math.cos(self.node.current_odom_xy_theta[2]),
            ),
        )
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        msg.color.r = 0.0
        msg.color.g = 0.0
        msg.color.b = 1.0
        msg.color.a = 1.0
        self.curr_dir_pub.publish(msg)
