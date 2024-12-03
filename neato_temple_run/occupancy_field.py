""" An implementation of an occupancy field that you can use to implement
    your particle filter """

import rclpy
import numpy as np
from sklearn.neighbors import NearestNeighbors

class OccupancyField(object):
    """ Stores an occupancy field for an input map.  An occupancy field returns
        the distance to the closest obstacle for any coordinate in the map
        Attributes:
            map: the map to localize against (nav_msgs/OccupancyGrid)
            closest_occ: the distance for each entry in the OccupancyGrid to
            the closest obstacle
    """

    def __init__(self, node):
        # grab the map
        self.map_width = 100
        self.map_height = 100
        self.map_resolution = 1
        self.map_origin_x = 0
        self.map_origin_y = 0
        self.total_size = self.map_width*self.map_height
        self.map_data = np.zeros((self.total_size))
        random_indices = np.int64(np.random.random_sample((10))*self.total_size)
        self.map_data[random_indices] = 1
        node.get_logger().info("map received width: {0} height: {1}".format(self.map_width, self.map_height))
        # The coordinates of each grid cell in the map
        X = np.zeros((self.map_width*self.map_height, 2))

        # while we're at it let's count the number of occupied cells
        total_occupied = 0
        curr = 0
        for i in range(self.map_width):
            for j in range(self.map_height):
                # occupancy grids are stored in row major order
                ind = i + j*self.map_width
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
                ind = i + j*self.map_width
                if self.map_data[ind] > 0:
                    occupied[curr, 0] = float(i)
                    occupied[curr, 1] = float(j)
                    curr += 1
        node.get_logger().info("building ball tree")
        # use super fast scikit learn nearest neighbor algorithm
        nbrs = NearestNeighbors(n_neighbors=1,
                                algorithm="ball_tree").fit(occupied)
        node.get_logger().info("finding neighbors")
        distances, indices = nbrs.kneighbors(X)

        node.get_logger().info("populating occupancy field")
        self.closest_occ = np.zeros((self.map_width, self.map_height))
        curr = 0
        for i in range(self.map_width):
            for j in range(self.map_height):
                self.closest_occ[i, j] = \
                    distances[curr][0]*self.map_resolution
                curr += 1
        self.occupied = occupied
        node.get_logger().info("occupancy field ready")

    def get_obstacle_bounding_box(self):
        """
        Returns: the upper and lower bounds of x and y such that the resultant
        bounding box contains all of the obstacles in the map.  The format of
        the return value is ((x_lower, x_upper), (y_lower, y_upper))
        """
        lower_bounds = self.occupied.min(axis=0)
        upper_bounds = self.occupied.max(axis=0)
        r = self.map_resolution
        return ((lower_bounds[0]*r + self.map_origin_x,
                 upper_bounds[0]*r + self.map_origin_x),
                (lower_bounds[1]*r + self.map_origin_y,
                 upper_bounds[1]*r + self.map_origin_y))

    def get_closest_obstacle_distance(self, x, y):
        """ Compute the closest obstacle to the specified (x,y) coordinate in
            the map.  If the (x,y) coordinate is out of the map boundaries, nan
            will be returned. """
        x_coord = (x - self.map_origin_x)/self.map_resolution
        y_coord = (y - self.map_origin_y)/self.map_resolution
        if type(x) is np.ndarray:
            x_coord = x_coord.astype(np.int)
            y_coord = y_coord.astype(np.int)
        else:
            x_coord = int(x_coord)
            y_coord = int(y_coord)

        is_valid = (x_coord >= 0) & (y_coord >= 0) & (x_coord < self.map_width) & (y_coord < self.map_height)
        if type(x) is np.ndarray:
            distances = np.float('nan')*np.ones(x_coord.shape)
            distances[is_valid] = self.closest_occ[x_coord[is_valid], y_coord[is_valid]]
            return distances
        else:
            return self.closest_occ[x_coord, y_coord] if is_valid else float('nan')
