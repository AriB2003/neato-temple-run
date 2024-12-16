""" An implementation of an occupancy field that you can use to implement
    your particle filter """

import rclpy
import time
import numpy as np
from builtin_interfaces.msg import Time
from nav_msgs.msg import OccupancyGrid
from sklearn.neighbors import NearestNeighbors
from sensor_msgs.msg import PointCloud
from threading import Thread


class OccupancyField(object):
    """Stores an occupancy field for an input map.  An occupancy field returns
    the distance to the closest obstacle for any coordinate in the map
    Attributes:
        map: the map to localize against (nav_msgs/OccupancyGrid)
        closest_occ: the distance for each entry in the OccupancyGrid to
        the closest obstacle
    """

    def __init__(self, node):
        self.node = node
        self.occ_pub = node.create_publisher(OccupancyGrid, "occupancy_grid", 10)
        self.timer = node.create_timer(0.1, self.publish_occupancy_grid)
        # grab the map
        map_size_m = 5
        # self.number_of_obstacles = 50
        self.map_resolution = 0.05
        self.map_width = int(map_size_m / self.map_resolution)
        self.map_height = int(map_size_m / self.map_resolution)
        self.map_origin_x = -self.map_width / 2 * self.map_resolution
        self.map_origin_y = -self.map_height / 2 * self.map_resolution
        self.total_size = self.map_width * self.map_height
        self.updated = False
        self.offset = 0.3

        self.node.create_subscription(
            PointCloud, "lidar_depth", self.process_points, 10
        )
        # self.timer = self.node.create_timer(0.1, self.process_points)
        self.map_indices = np.array([])
        self.closest_occ = 100 * np.ones((self.map_width, self.map_height))

        thread = Thread(target=self.loop_wrapper)
        thread.start()

    def loop_wrapper(self):
        while True:
            self.build()
            time.sleep(0.25)

    def build(self):
        if self.map_indices.shape[0] == 0:
            print("No obstacles")
            return
        print("Obstacles")
        indices = (
            self.map_indices - np.array([self.map_origin_x, self.map_origin_y])
        ) // self.map_resolution
        indices = np.maximum(indices, [0, 0])
        indices = np.minimum(indices, [self.map_width - 1, self.map_height - 1])
        linearized = np.int64(indices[:, 0] * self.map_height + indices[:, 1])

        self.map_data = np.zeros((self.total_size))
        # random_indices = np.int64(np.random.random_sample((np.random.random_integers(1,100)))*self.total_size)
        # random_indices = np.int64(
        #     np.random.random_sample(self.number_of_obstacles) * self.total_size
        # )
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
        self.closest_occ = np.zeros((self.map_width, self.map_height))
        curr = 0
        for i in range(self.map_width):
            for j in range(self.map_height):
                self.closest_occ[i, j] = distances[curr][0] * self.map_resolution
                curr += 1
        self.occupied = occupied
        self.node.get_logger().info("occupancy field ready")
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

    def get_index_from_coords(self, x, y):
        x_coord = (x - self.map_origin_x) / self.map_resolution
        y_coord = (y - self.map_origin_y) / self.map_resolution
        if type(x) is np.ndarray:
            x_coord = np.round(x_coord).astype(np.int)
            y_coord = np.round(y_coord).astype(np.int)
        else:
            x_coord = int(round(x_coord))
            y_coord = int(round(y_coord))

        is_valid = (
            (x_coord >= 0)
            & (y_coord >= 0)
            & (x_coord < self.map_width)
            & (y_coord < self.map_height)
        )
        return x_coord, y_coord, is_valid

    def get_coords_from_index(self, idx, idy):
        x = idx * self.map_resolution + self.map_origin_x
        y = idy * self.map_resolution + self.map_origin_y
        is_valid = (
            (idx >= 0)
            & (idy >= 0)
            & (idx < self.map_width * self.map_resolution)
            & (idy < self.map_height * self.map_resolution)
        )
        return x, y, is_valid

    def get_closest_obstacle_distance(self, x, y):
        """Compute the closest obstacle to the specified (x,y) coordinate in
        the map.  If the (x,y) coordinate is out of the map boundaries, nan
        will be returned."""
        x_coord, y_coord, is_valid = self.get_index_from_coords(x, y)
        if type(x) is np.ndarray:
            distances = np.float("nan") * np.ones(x_coord.shape)
            distances[is_valid] = self.closest_occ[x_coord[is_valid], y_coord[is_valid]]
            return distances
        else:
            return self.closest_occ[x_coord, y_coord] if is_valid else float("nan")

    def publish_occupancy_grid(self):
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

    def process_points(self, msg: PointCloud):
        print("Points received")
        self.map_indices = np.array([[p.x, p.y] for p in msg.points])
