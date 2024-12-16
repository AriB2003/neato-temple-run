"""
Requires
- transformers
- torch

"""

import time
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image, ImageOps
import cv2

def estimate(image, width, height):
    """
    Processes image,
    Take a PIL image input
    """

    image_processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )

    # prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # interpolate to original size and visualize the prediction
    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(height, width)],
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]
    depth = (predicted_depth - predicted_depth.min()) / (
        predicted_depth.max() - predicted_depth.min()
    )

    depth = depth.numpy()

    # depth = depth.detach().cpu().numpy() * 255
    # depth = Image.fromarray(depth.astype("uint8"))
    # depth = ImageOps.colorize(
    #     depth, black="black", white="orange", mid="purple", blackpoint=20
    # )
    # # can set <whitepoint>  and <midpoint> as well. This lets us calibrate ?
    return depth


def cv2_wrapper(img, new_width):
    """
    Converts cv2 image to PIL image for image processing input
    """
    height, width, channels = img.shape
    img = cv2.resize(
        img, (new_width, height * new_width // width), interpolation=cv2.INTER_AREA
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_raw = Image.fromarray(img)

    return estimate(image_raw, width, height)


# testing purposes
if __name__ == "__main__":

    # - basketball.JPG
    # - pedestrian.jpeg
    # - backpack_2.jpg
    # - skyline.jpg
    img = cv2.imread("neato_temple_run/Where-We-Fly.jpg")

    t0 = time.time()

    depth = cv2_wrapper(img, 512)  # 64 is fastest

    t1 = time.time()
    print(t1 - t0)
    # depth.show()

    import matplotlib.pyplot as plt

    plt.imshow(
        depth,
        cmap=plt.get_cmap("inferno"),
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    plt.axis("off")
    plt.show()
