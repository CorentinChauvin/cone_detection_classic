#!/usr/bin/env python2.7
"""
    Attempt to detect cones with classical computer vision

    execfile('cone_segmentation.py')
"""

import rospkg
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from time import time

# Tuning parameters
min_nbr_pixels = 30     # min nbr of pixels that a cone must have
max_nbr_pixels = 3000   # max nbr of pixels that a cone must have
min_aspect_ratio = 0.7  # min width/height that a cluster must have
max_aspect_ratio = 2.0  # max width/height that a cluster must have
lim_aspect_ratio = 1.0  # below this, top of the cone ; above, base of the cone
group_percent = 0.1     # percentage of lateral deviation between top and base of cone to group it
blue_min = (100, 100, 100)
blue_max = (130, 255, 255)
yellow_min = (20, 100, 100)
yellow_max = (40, 255, 255)

verbose = False


def is_base_cone(aspect_ratio):
    """ Check whether the cluster is the base of the cone considering the ratio widh/height
    """
    return aspect_ratio >= lim_aspect_ratio and aspect_ratio <= max_aspect_ratio


def is_top_cone(aspect_ratio):
    """ Check whether the cluster is the top of the cone considering the ratio widh/height
    """
    return aspect_ratio <= lim_aspect_ratio and aspect_ratio >= min_aspect_ratio


# Load the image
rospack = rospkg.RosPack()
rospack.list()
path = rospack.get_path('detection_classic')
filename = path + "/images/1.png"
img = cv2.imread(filename)
# img = cv2.resize(img, (640, 360))
(height, width, _) = img.shape

init_time = time()


def find_bouding_boxes(min_hsv, max_hsv):
    """
        Find the cones in the image for the given range of hsv colors

        Arguments: range of HSV colors for segmentation
        Returns: a list of bounding boxes corresponding to cones
    """

    # HSV segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, min_hsv, max_hsv)
    img_segmented = cv2.bitwise_and(img, img, mask=mask)

    # cv2.imshow('image', img_segmented)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Connected components clustering
    outputs = cv2.connectedComponentsWithStats(mask, 8)
    nbr_clusters = outputs[0]
    labels = outputs[1]
    stats = outputs[2]
    centroids = outputs[3]
    kept_label = [True for k in range(nbr_clusters)]  # whether the cluster is kept

    print("Intermediary time: {}".format(time() - init_time))

    # Select clusters based on geometry
    for k in range(nbr_clusters):
        area = stats[k][cv2.CC_STAT_AREA]
        w = stats[k][cv2.CC_STAT_WIDTH]
        h = stats[k][cv2.CC_STAT_HEIGHT]
        x = centroids[k][1]
        y = centroids[k][0]
        aspect_ratio = w/float(h)

        if (area < min_nbr_pixels or area > max_nbr_pixels
            or aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio):
            kept_label[k] = False

    print("Intermediary time 2: {}".format(time() - init_time))

    # Debug only
    if verbose:
        for k in range(nbr_clusters):
            if kept_label[k]:
                area = stats[k][cv2.CC_STAT_AREA]
                w = stats[k][cv2.CC_STAT_WIDTH]
                h = stats[k][cv2.CC_STAT_HEIGHT]
                x = centroids[k][1]
                y = centroids[k][0]
                aspect_ratio = w/float(h)
                print("area={:.0f} ; size=({:.0f}, {:.0f}) ; center=({:.0f}, {:.0f}) ; ratio={:.2f}".format(area, h, w, x, y, aspect_ratio))

    # Group top and bottom of cones (two clusters sometimes)
    groups = []  # groups of one or two clusters belonging to the same cone
    label_grouped = [False]*nbr_clusters
    for k in range(nbr_clusters):
        if kept_label[k]:
            w1 = stats[k][cv2.CC_STAT_WIDTH]
            h1 = stats[k][cv2.CC_STAT_HEIGHT]
            aspect_ratio = w1/float(h1)
            y1 = centroids[k][0]
            min_y = y1 - group_percent*w1
            max_y = y1 + group_percent*w1
            group_found = False

            for l in range(k+1, nbr_clusters):
                y2 = centroids[l][0]
                if kept_label[l] and y2 >= min_y and y2 <= max_y:
                    groups.append([k, l])
                    label_grouped[k] = True
                    label_grouped[l] = True
                    group_found = True
                    break
            if not group_found and not label_grouped[k]:
                groups.append([k])

    print("Intermediary time 3: {}".format(time() - init_time))

    # Find bouding boxes around the clusters
    bounding_boxes = []
    for group in groups:
        pixels_list = []

        for k in group:
            pxl = np.argwhere(labels == k)
            pixels_list.extend(pxl.tolist())

        pixels_array = np.array(pixels_list)
        bb = cv2.boundingRect(pixels_array)

        if len(group) == 1:
            k = group[0]
            h = stats[k][cv2.CC_STAT_HEIGHT]
            H = 2.5 * h
            bb = (centroids[k][1] - H + h/2, bb[1], H, bb[3])

        bounding_boxes.append(bb)

    print("Intermediary time 4: {}".format(time() - init_time))



    # Display segmentation
    # clustered_image = np.zeros((height, width))
    # for i in range(height):
    #     for j in range(width):
    #         if kept_label[labels.item(i, j)]:
    #             clustered_image[i][j] = 255
    # plt.imshow(clustered_image)
    # plt.show()

    print("---")
    print(bounding_boxes)


    return bounding_boxes


blue_boxes = find_bouding_boxes(blue_min, blue_max)
yellow_boxes = find_bouding_boxes(yellow_min, yellow_max)

print("Total time: {}".format(time() - init_time))



# Plotting the results
fig, ax = plt.subplots(1)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)
for bb in blue_boxes:
    rect = Rectangle((bb[1], bb[0]), bb[3], bb[2],
                     linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)
for bb in yellow_boxes:
    rect = Rectangle((bb[1], bb[0]), bb[3], bb[2],
                     linewidth=1, edgecolor='y', facecolor='none')
    ax.add_patch(rect)
plt.show()
