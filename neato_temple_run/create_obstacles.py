"""
takes a opencv and LIDAR message
"""

import time
import numpy as np
from PIL import Image, ImageOps
import cv2
from depth_estimation_single_photo import cv2_wrapper

horizon_y = 190  # 233
threshold = 0.2  # .2


def remove_depth_floor(cv2_image):

    h, w = cv2_image.shape

    min_value = 0
    max_value = np.mean(cv2_image[-3:, :])

    horizon = h - horizon_y
    btm_gradient = np.linspace(min_value, max_value, horizon, endpoint=True)
    btm_gradient = np.tile(btm_gradient, (w, 1))
    btm_gradient = np.transpose(btm_gradient)

    btm_matte = np.full((h - btm_gradient.shape[0], w), 1)
    btm = np.concatenate((btm_matte, btm_gradient))

    depth_remove_floor = np.subtract(cv2_image, btm)

    return depth_remove_floor


def identify_obstacles(no_floor, depth_map):
    s = depth_map.shape

    mask = np.zeros(s, dtype="uint8")
    mask[no_floor < threshold] = 0
    mask[no_floor > threshold] = 1

    masked_map = cv2.bitwise_and(depth_map, depth_map, mask=mask)

    return masked_map, mask


def generate_point_cloud(depth_map, mask):
    """
    Takes masked_map and depth_map
    """
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # may end up removing
    distances = []
    projected_width = []

    # only list still used
    cartesian_points = []

    height, width = mask.shape

    for contour in contours:

        blank = np.zeros((height, width), dtype=np.uint8)

        cv2.drawContours(blank, [contour], 0, 255, -1)
        mean = cv2.mean(depth_map, mask=blank)[0]
        distance = 29.61984 * 1 / mean - 9.58096
        x, y, w, h = cv2.boundingRect(contour)

        """remove from here"""
        # width of a 3 inch object
        width_3in = 3186.06643 / distance + 2.26254

        # ratio of width to 3 inch object is used to find real width
        # in inches
        width_ratio = w / width_3in * 3 * 2.54
        projected_width += [width_ratio]

        distances += [distance]

        """TO here"""

        coords = [(x + i, y + h, 1 / 6) for i in range(0, w, 10)]

        coords_np = np.array(coords)
        rays = find_angle(coords_np)
        rays[:, 1] = 0
        norm_ratio = distance / np.linalg.norm(rays, axis=1)
        projected_coords = np.multiply(norm_ratio[:, np.newaxis], rays)

        cartesian_points += np.array(projected_coords).tolist()

        # for spacer in range(0, w, 10):
        #     ray1 = np.array(find_angle(x + spacer, y + h))
        #     ray1_projected = ray1
        #     ray1_projected[1] = 0
        #     ray1_projected = (
        #         np.linalg.norm(ray1) / np.linalg.norm(ray1_projected) * ray1_projected
        #     )
        #     cartesian_points += [
        #         distance / np.linalg.norm(ray1_projected) * ray1_projected
        #     ]

        # ray2 = np.array(find_angle(x + w, y + h))
        # cartesian_points += [distance / np.linalg.norm(ray2) * ray2]

        # # cartesian points projects the width from the camera, which doesn't follow the width prediction

    isNull = False
    if not cartesian_points:
        isNull = True
    cartesian_points = np.array(cartesian_points)

    # print(distances)
    # print(projected_width)

    return cartesian_points, isNull


def find_angle(coords_np):

    width = 4608
    height = 2592
    f = 2.75e-3 / 1.4e-6

    coords_np *= 6

    cx = (width - 1) / 2
    cy = (height - 1) / 2

    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    Ki = np.linalg.inv(K)
    r1 = coords_np.dot(np.transpose(Ki))  # ray to object
    return r1


# def find_angle(x, y):

#     width = 4608
#     height = 2592
#     f = 2.75e-3 / 1.4e-6

#     x *= 6
#     y *= 6

#     cx = (width - 1) / 2
#     cy = (height - 1) / 2

#     K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
#     Ki = np.linalg.inv(K)
#     r1 = Ki.dot([x, y, 1.0])  # ray to object
#     return r1


# testing purposes
if __name__ == "__main__":

    # - Figure 1 - 120
    # - Figure 2 - 90
    # - Figure 3 - 70
    # - Figure 4 - 53
    # - Figure 5 - 20
    # - Figure 6 - 35
    img = cv2.imread("Figure_4.png")
    depth = cv2_wrapper(img, 64)  # 64 is fastest

    t0 = time.time()

    depth_remove_floor = remove_depth_floor(depth)
    masked_map, mask = identify_obstacles(depth_remove_floor, depth)

    cartesian_points, width = generate_point_cloud(depth, mask)

    t1 = time.time()
    print(t1 - t0)
    # depth.show()
    import matplotlib.pyplot as plt

    figs, axs = plt.subplots(2, 2)

    axs[0][0].imshow(
        depth,
        cmap=plt.get_cmap("hot"),
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    axs[0][0].axis("off")

    axs[0][1].imshow(
        depth_remove_floor,
        cmap=plt.get_cmap("hot"),
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    axs[0][1].axis("off")

    axs[1][1].scatter(cartesian_points[:, 0], cartesian_points[:, 2])
    axs[1][1].set_xlim([-90, 90])
    axs[1][1].set_ylim([0, 130])

    axs[1][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[1][0].axis("off")

    plt.show()
