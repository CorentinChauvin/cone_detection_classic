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
min_nbr_pixels = 30      # min nbr of pixels that a cone must have
max_nbr_pixels = 8000    # max nbr of pixels that a cone must have
min_cluster_ratio = 0.4  # min width/height that a cluster must have
max_cluster_ratio = 2.0  # max width/height that a cluster must have
lim_cluster_ratio = 1.0  # below this, top of the cone ; above, base of the cone
min_box_ratio = 0.6      # min width/height that a box must have
max_box_ratio = 0.9      # max width/height that a box must have
group_percent = 0.3      # percentage of lateral deviation between top and base of cone to group it
blue_min = (100, 100, 100)
blue_max = (130, 255, 255)
yellow_min = (15, 100, 200)
yellow_max = (30, 255, 255)
gaussian_size = 5

invert = True
gaussian_blur = False
hist_equalisation = False
verbose = True


def is_base_cone(cluster_ratio):
    """ Check whether the cluster is the base of the cone considering the ratio widh/height
    """
    return cluster_ratio >= lim_cluster_ratio and cluster_ratio <= max_cluster_ratio


def is_top_cone(cluster_ratio):
    """ Check whether the cluster is the top of the cone considering the ratio widh/height
    """
    return cluster_ratio <= lim_cluster_ratio and cluster_ratio >= min_cluster_ratio


def find_bouding_boxes(img, min_hsv, max_hsv):
    """
        Find the cones in the image for the given range of hsv colors

        Arguments:
        - img: input image
        - min_hsv, max_hsv: range of HSV colors for segmentation
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

    # Select clusters based on geometry
    for k in range(nbr_clusters):
        area = stats[k][cv2.CC_STAT_AREA]
        w = stats[k][cv2.CC_STAT_WIDTH]
        h = stats[k][cv2.CC_STAT_HEIGHT]
        x = centroids[k][1]
        y = centroids[k][0]
        cluster_ratio = w/float(h)

        if (area < min_nbr_pixels or area > max_nbr_pixels
            or cluster_ratio < min_cluster_ratio or cluster_ratio > max_cluster_ratio):
            kept_label[k] = False

    # Debug only
    if verbose:
        for k in range(nbr_clusters):
            area = stats[k][cv2.CC_STAT_AREA]

            if area > min_nbr_pixels:
                w = stats[k][cv2.CC_STAT_WIDTH]
                h = stats[k][cv2.CC_STAT_HEIGHT]
                x = centroids[k][1]
                y = centroids[k][0]
                cluster_ratio = w/float(h)
                kept = 'y' if kept_label[k] else 'n'
                print("[{}] area={:.0f} ; size=({:.0f}, {:.0f}) ; center=({:.0f}, {:.0f}) ; ratio={:.2f}".format(kept, area, h, w, x, y, cluster_ratio))

    # Group top and bottom of cones (two clusters sometimes)
    groups = []  # groups of one or two clusters belonging to the same cone
    label_grouped = [False]*nbr_clusters

    for k in range(nbr_clusters-1, -1, -1):
        # Assumption: the clustered are ordered (first ones at the top of the image)

        if kept_label[k]:
            w1 = stats[k][cv2.CC_STAT_WIDTH]
            h1 = stats[k][cv2.CC_STAT_HEIGHT]
            cluster_ratio = w1/float(h1)
            y1 = centroids[k][0]
            min_y = y1 - group_percent*w1
            max_y = y1 + group_percent*w1
            # min_x =
            group_found = False

            for l in range(k-1, -1, -1):
                y2 = centroids[l][0]
                if kept_label[l] and y2 >= min_y and y2 <= max_y:
                    groups.append([k, l])
                    label_grouped[k] = True
                    label_grouped[l] = True
                    group_found = True
                    break
            if not group_found and not label_grouped[k]:
                groups.append([k])

    # Find bouding boxes around the clusters
    bounding_boxes = []
    for group in groups:
        if len(group) == 1:
            # Only the base of the cone has been found
            k = group[0]
            h = stats[k][cv2.CC_STAT_HEIGHT]
            H = 2.5 * h
            W = stats[k][cv2.CC_STAT_WIDTH]
            c_x = centroids[k][1] - H + h/2.0
            c_y = centroids[k][0] - W/2.0

            bb = (c_x, c_y, H, W)
            # bounding_boxes.append(bb)
        elif len(group) == 2:
            # Both the base and the top of the cone have been found
            k = group[0]
            l = group[1]
            H = abs(centroids[k][1] - centroids[l][1]) + (stats[k][cv2.CC_STAT_HEIGHT] + stats[l][cv2.CC_STAT_HEIGHT])/2.0
            W = max(stats[k][cv2.CC_STAT_WIDTH], stats[l][cv2.CC_STAT_WIDTH])
            c_x = (centroids[k][1] + centroids[l][1])/2.0 - H/2.0
            c_y = (centroids[k][0] + centroids[l][0])/2.0 - W/2.0

            bb = (c_x, c_y, H, W)
            ratio = W / max(1.0, float(H))

            if ratio >= min_box_ratio and ratio <= max_box_ratio:
                bounding_boxes.append(bb)
            elif verbose:
                print("[n] Box (W, H)=({:.0f}, {:.0f}) ; corner=({:.0f}, {:.0f}) ; ratio={:.2f}".format(bb[2], bb[3], bb[0], bb[1], ratio))


    # Display segmentation
    # clustered_image = np.zeros((height, width))
    # for i in range(height):
    #     for j in range(width):
    #         if kept_label[labels.item(i, j)]:
    #             clustered_image[i][j] = 255
    # plt.imshow(clustered_image)
    # plt.show()

    if verbose:
        for bb in bounding_boxes:
            ratio = bb[3] / max(1.0, float(bb[2]))
            print("[y] Box (W, H)=({:.0f}, {:.0f}) ; corner=({:.0f}, {:.0f}) ; ratio={:.2f}".format(bb[2], bb[3], bb[0], bb[1], ratio))

    return bounding_boxes


def plot_results(img, blue_boxes, yellow_boxes):
    """ Plot the image and the bounding boxes corresponding to the cones

        Arguments:
        - img: source image
        - blue_boxes: bounding boxes of the blue cones
        - yellow_boxes: bounding boxes of the yellow cones
    """

    fig, ax = plt.subplots(1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    for bb in blue_boxes:
        rect = Rectangle((bb[1], bb[0]), bb[3], bb[2],
                        linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    for bb in yellow_boxes:
        rect = Rectangle((bb[1], bb[0]), bb[3], bb[2],
                        linewidth=2, edgecolor='y', facecolor='none')
        ax.add_patch(rect)
    plt.show()


def generate_output(img, blue_boxes, yellow_boxes):
    """ Generate an openCV output image

        Arguments:
        - img: source image
        - blue_boxes: bounding boxes of the blue cones
        - yellow_boxes: bounding boxes of the yellow cones
        Returns:
        - img: output image
    """

    if invert:
        img = cv2.bitwise_not(img)

    for box in blue_boxes:
        bb = [int(box[k]) for k in range(len(box))]
        pt1 = (bb[1], bb[0])
        pt2 = (bb[1] + bb[3], bb[0] + bb[2])
        cv2.rectangle(img, pt1, pt2, color=(255, 0, 0), thickness=3)
    for box in yellow_boxes:
        bb = [int(box[k]) for k in range(len(box))]
        pt1 = (bb[1], bb[0])
        pt2 = (bb[1] + bb[3], bb[0] + bb[2])
        cv2.rectangle(img, pt1, pt2, color=(0, 255, 255), thickness=3)

    return img


def preprocess(img):
    """ Preprocess image to enhance detection
    """

    global blue_min, blue_max, yellow_min, yellow_max

    if invert:
        img = cv2.bitwise_not(img)

    if hist_equalisation:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    if gaussian_blur:
        n = gaussian_size
        kernel = np.ones((n, n), np.float32) / n**2
        img = cv2.filter2D(img, -1, kernel)

    return img



if False:
    # Load the image
    rospack = rospkg.RosPack()
    rospack.list()
    path = rospack.get_path('detection_classic')
    file_name = path + "/images/7.png"
    img = cv2.imread(file_name)
    (height, width, _) = img.shape

    # Image enhancement
    if invert:
        (blue_min, blue_max, yellow_min, yellow_max) = (yellow_min, yellow_max, blue_min, blue_max)
    img = preprocess(img)

    # Find the cones
    init_time = time()
    blue_boxes = find_bouding_boxes(img, blue_min, blue_max)
    print("---")
    yellow_boxes = find_bouding_boxes(img, yellow_min, yellow_max)
    print("Total time: {}".format(time() - init_time))

    # Plot the results
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv[:, :, 1] = hsv[:, :, 0]
    # hsv[:, :, 2] = hsv[:, :, 0]
    # if invert:
    #     img = cv2.bitwise_not(img)
    plot_results(img, blue_boxes, yellow_boxes)

else:
    # Load the video
    rospack = rospkg.RosPack()
    rospack.list()
    path = rospack.get_path('detection_classic')
    file_name = path + "/images/kth.mp4"
    video = cv2.VideoCapture(file_name)

    if invert:
        (blue_min, blue_max, yellow_min, yellow_max) = (yellow_min, yellow_max, blue_min, blue_max)


    while video.isOpened():
        ret, frame = video.read()

        frame = preprocess(frame)
        blue_boxes = find_bouding_boxes(frame, blue_min, blue_max)
        yellow_boxes = find_bouding_boxes(frame, yellow_min, yellow_max)

        output = generate_output(frame, blue_boxes, yellow_boxes)
        cv2.imshow('test', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
