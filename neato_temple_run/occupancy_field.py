""" An implementation of an occupancy field """

import time
import math
import numpy as np
from builtin_interfaces.msg import Time
from nav_msgs.msg import OccupancyGrid
from sklearn.neighbors import NearestNeighbors
from sensor_msgs.msg import PointCloud
from threading import Thread


class OccupancyField(object):
    """ Stores an occupancy field for an input map.  An occupancy field returns
    the distance to the closest obstacle for any coordinate in the map """

    def __init__(self, node):
        self.node = node # NeatoRunner node

        # publisher for the grid
        self.occ_pub = node.create_publisher(OccupancyGrid, "occupancy_grid", 10)
        self.timer = node.create_timer(0.1, self.publish_occupancy_grid)

        map_size_m = 20 # map size [m]
        self.map_resolution = 0.05 # map resolution [m/idx]
        self.map_width = int(map_size_m / self.map_resolution) # map width [idx]
        self.map_height = int(map_size_m / self.map_resolution) # map height [idx]
        self.map_origin_x = -self.map_width / 2 * self.map_resolution # map origin x [m]
        self.map_origin_y = -self.map_height / 2 * self.map_resolution # map origin y [m]
        self.total_size = self.map_width * self.map_height # map size [idx^2]
        self.updated = False # update flag
        self.offset = 0.3 # obstacle offset for visualization

        # subscribers for lidar and vision pipeline
        self.node.create_subscription(
            PointCloud, "lidar_depth", self.process_lidar_points, 10
        )
        self.node.create_subscription(
            PointCloud, "monocular_depth", self.process_mono_points, 10
        )

        # obstacle indicies
        self.lidar_coords = np.array([])
        self.mono_coords = np.array([])
        self.obs_coords = np.array([])

        # obstacle distances
        self.closest_occ = np.ones((self.map_width, self.map_height))

        # parallelism thread
        thread = Thread(target=self.loop_wrapper)
        thread.start()

    def loop_wrapper(self):
        while True:
            # re-build once a second
            self.build()
            time.sleep(1)

    def build(self):
        """ Re-build the occupancy grid with the latest horizon data """
        # if no data, return
        if self.lidar_coords.shape[0] == 0:
            return
        # if no optical data, use only lidar
        if self.mono_coords.shape[0] == 0:
            self.obs_coords = self.lidar_coords
        else:
            # use only lidar or monocular with lidar (depending on environment)
            self.obs_coords = self.lidar_coords
            # self.obs_coords = np.vstack((self.lidar_coords, self.mono_coords))

        # extract the indices from the obstacles
        indices = (
            self.obs_coords - np.array([self.map_origin_x, self.map_origin_y])
        ) // self.map_resolution
        # bound the indices to the occupancy grid
        indices = np.maximum(indices, [0, 0])
        indices = np.minimum(indices, [self.map_width - 1, self.map_height - 1])
        # linearize the indices
        linearized = np.int64(indices[:, 0] * self.map_height + indices[:, 1])

        # create the map data and add the obstacle locations
        self.map_data = np.zeros((self.total_size))
        self.map_data[linearized] = 1
        self.node.get_logger().info(
            "map received width: {0} height: {1}".format(
                self.map_width, self.map_height
            )
        )
        # The coordinates of each grid cell in the map
        X = np.zeros((self.map_width * self.map_height, 2))

        # while we're at it let's count the number of occupied cells
        total_occupied = 0
        curr = 0
        for i in range(self.map_width):
            for j in range(self.map_height):
                # occupancy grids are stored in row major order
                ind = i + j * self.map_width
                if self.map_data[ind] > 0:
                    total_occupied += 1
                X[curr, 0] = float(i)
                X[curr, 1] = float(j)
                curr += 1

        # The coordinates of each occupied grid cell in the map
        occupied = np.zeros((total_occupied, 2))
        curr = 0
        for i in range(self.map_width):
            for j in range(self.map_height):
                # occupancy grids are stored in row major order
                ind = i + j * self.map_width
                if self.map_data[ind] > 0:
                    occupied[curr, 0] = float(i)
                    occupied[curr, 1] = float(j)
                    curr += 1
        self.node.get_logger().info("building ball tree")
        # use super fast scikit learn nearest neighbor algorithm
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(occupied)
        self.node.get_logger().info("finding neighbors")
        distances, indices = nbrs.kneighbors(X)

        self.node.get_logger().info("populating occupancy field")
        temp_closest_occ = np.zeros((self.map_width, self.map_height))
        curr = 0
        for i in range(self.map_width):
            for j in range(self.map_height):
                temp_closest_occ[i, j] = distances[curr][0] * self.map_resolution
                curr += 1
        self.occupied = occupied
        self.node.get_logger().info("occupancy field ready")

        # save occupancy grid and trigger update
        self.closest_occ = temp_closest_occ
        self.updated = True

    def get_obstacle_bounding_box(self):
        """
        Returns: the upper and lower bounds of x and y such that the resultant
        bounding box contains all of the obstacles in the map.  The format of
        the return value is ((x_lower, x_upper), (y_lower, y_upper))
        """
        lower_bounds = self.occupied.min(axis=0)
        upper_bounds = self.occupied.max(axis=0)
        r = self.map_resolution
        return (
            (
                lower_bounds[0] * r + self.map_origin_x,
                upper_bounds[0] * r + self.map_origin_x,
            ),
            (
                lower_bounds[1] * r + self.map_origin_y,
                upper_bounds[1] * r + self.map_origin_y,
            ),
        )

    def get_index_from_coords(self, coord_x, coord_y):
        """ Get the index of a point from the coordinates """
        x_ind = (coord_x - self.map_origin_x) // self.map_resolution
        y_ind = (coord_y - self.map_origin_y) // self.map_resolution
        if type(coord_x) is np.ndarray:
            x_ind = np.round(x_ind).astype(np.int)
            y_ind = np.round(y_ind).astype(np.int)
        else:
            x_ind = int(round(x_ind))
            y_ind = int(round(y_ind))
        # check valid bounds
        is_valid = (
            (x_ind >= 0)
            & (y_ind >= 0)
            & (x_ind < self.map_width)
            & (y_ind < self.map_height)
        )
        return x_ind, y_ind, is_valid

    def get_coords_from_index(self, idx, idy):
        """ Get the coordinates of a point from the index """
        x = idx * self.map_resolution + self.map_origin_x
        y = idy * self.map_resolution + self.map_origin_y
        # check valid bounds
        is_valid = (
            (idx >= 0)
            & (idy >= 0)
            & (idx < self.map_width)
            & (idy < self.map_height)
        )
        return x, y, is_valid

    def get_closest_obstacle_distance(self, x, y):
        """ Compute the closest obstacle to the specified (x,y) coordinate in
        the map. If the (x,y) coordinate is out of the map boundaries, -infinity
        will be returned """
        x_coord, y_coord, is_valid = self.get_index_from_coords(x, y)
        return self.closest_occ[x_coord, y_coord] if is_valid else -math.inf

    def publish_occupancy_grid(self):
        """ Publish the occupancy grid """
        msg = OccupancyGrid()
        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.header.stamp = Time()
        msg.header.frame_id = "odom"
        msg.data = np.reshape(
            (100 * (self.closest_occ <= self.offset)).astype("int8"), -1
        ).tolist()
        self.occ_pub.publish(msg)

    def process_lidar_points(self, msg: PointCloud):
        """ Receive the lidar points """
        # print("Points received")
        self.lidar_coords = np.array([[p.x, p.y] for p in msg.points])

    def process_mono_points(self, msg: PointCloud):
        """ Receive the vision points """
        # print("Points received")
        self.mono_coords = np.array([[p.x, p.y] for p in msg.points])
