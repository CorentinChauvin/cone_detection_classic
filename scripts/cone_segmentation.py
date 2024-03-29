#!/usr/bin/env python2.7
"""
    Attempt to detect cones with classical computer vision

    Corentin Chauvin-Hameau - KTH Formula Student 2019

    TODO:
    - Only do segmentation in the center of the image for middle stripe searching
    - Take into account the size of the cones with respect to the x coordinate
"""

import rospkg
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from time import time
from copy import deepcopy

# Tuning parameters
min_nbr_pixels = 50      # min nbr of pixels that a cone must have
max_nbr_pixels = 10000   # max nbr of pixels that a cone must have
min_cluster_ratio = 0.4  # min width/height that a cluster must have
max_cluster_ratio = 2.0  # max width/height that a cluster must have
lim_cluster_ratio = 1.0  # below this, top of the cone ; above, base of the cone
min_box_ratio = 0.3      # min width/height that a box must have
max_box_ratio = 1.5      # max width/height that a box must have
group_percent = 0.3      # percentage of lateral deviation between top and base of cone to group it
center_percent = 0.3     # threshold to consider a cluster to be in the center of the box
gaussian_size = 2        # size of the kernel for gaussian blur
bottom_ratio = 0.7       # ratio of the image to be kept in the vertical axis (only the bottom of the image is processed) (1.0=keep everything)

blue_min = (100, 100, 100)
blue_max = (140, 255, 255)
yellow_min = (10, 80, 200)
yellow_max = (25, 255, 255)

white_min = (0, 0, 200)
white_max = (255, 30, 255)
black_min = (0, 0, 0)
black_max = (50, 255, 180)

invert = True
gaussian_blur = False
hist_equalisation = False
verbose = False
save_video = True
save_image = True


def is_base_cone(cluster_ratio):
    """ Check whether the cluster is the base of the cone considering the ratio widh/height
    """
    return cluster_ratio >= lim_cluster_ratio and cluster_ratio <= max_cluster_ratio


def is_top_cone(cluster_ratio):
    """ Check whether the cluster is the top of the cone considering the ratio widh/height
    """
    return cluster_ratio <= lim_cluster_ratio and cluster_ratio >= min_cluster_ratio


def find_bouding_boxes(img, min_main, max_main, min_center, max_center):
    """
        Find the cones in the image for the given range of hsv colors

        Arguments:
        - img: input image
        - min_main, max_main: range of HSV colors for segmentation of base and top of cone
        - min_center, max_center: range of HSV colors for segmentation of center stripe
        Returns:
        - list of bounding boxes corresponding to cones
    """

    # Keep only the bottom part of the image
    (init_height, init_width, _) = img.shape
    x1 = (1-bottom_ratio) * init_height
    x2 = init_height
    img = img[int(x1):int(x2), :, :]
    (height, width, _) = img.shape

    # HSV segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, min_main, max_main)
    img_segmented = cv2.bitwise_and(img, img, mask=mask)

    # cv2.imshow('image', img_segmented)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # mask2 = cv2.inRange(hsv, min_center, max_center)
    # segmented = cv2.bitwise_and(img, img, mask=mask2)
    # cv2.imshow('image', mask2)
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

    # Check for the stripe in the center of the cone
    bb_tmp = deepcopy(bounding_boxes)
    bounding_boxes = []
    for bb in bb_tmp:
        x1 = int(max(0, bb[0]))
        x2 = int(min(height-1, x1 + bb[2]))
        y1 = int(max(0, bb[1]))
        y2 = int(min(width-1, y1 + bb[3]))

        # TODO: only do segmentation in the center of the image (reducing computations)
        sub_img = hsv[x1:x2+1, y1:y2+1, :]
        mask = cv2.inRange(sub_img, min_center, max_center)
        segmented = cv2.bitwise_and(sub_img, sub_img, mask=mask)

        outputs = cv2.connectedComponentsWithStats(mask, 8)
        nbr_clusters = outputs[0]
        stats = outputs[2]
        centroids = outputs[3]

        if verbose:
            print("---")

        center_found = False
        max_area = 0  # area corresponding to the background
        for k in range(nbr_clusters):
            if stats[k][cv2.CC_STAT_AREA] > max_area:
                max_area = stats[k][cv2.CC_STAT_AREA]

        for k in range(nbr_clusters):
            area = stats[k][cv2.CC_STAT_AREA]
            x = int(centroids[k][1])
            y = int(centroids[k][0])
            min_x = (0.4 - center_percent) * (x2 - x1)
            max_x = (0.4 + center_percent) * (x2 - x1)
            min_y = (0.5 - center_percent) * (y2 - y1)
            max_y = (0.5 + center_percent) * (y2 - y1)

            if verbose:
                print("area={:.0f} ; (x, y)=({:.0f}, {:.0f}) ; bx=({:.0f}, {:.0f}) ; by=({:.0f}, {:.0f})".format(
                    area, x, y, min_x, max_x, min_y, max_y))

            if (area != max_area and area >= min_nbr_pixels
                and x >= min_x and x <= max_x and y >= min_y and y <= max_y):
                center_found = True
                break

        if center_found:
            bounding_boxes.append(bb)
        elif verbose:
            print('rejected')

        # plt.subplot(121)
        # rgb = cv2.cvtColor(img[x1:x2+1, y1:y2+1, :], cv2.COLOR_BGR2RGB)
        # plt.imshow(rgb)
        # plt.subplot(122)
        # plt.imshow(mask)
        # plt.show()

    # Offset the bounding boxes to take into account the neglicted top part
    bb_tmp = deepcopy(bounding_boxes)
    bounding_boxes = []
    offset_x = int((1-bottom_ratio) * init_height)
    for bb in bb_tmp:
        bounding_boxes.append((bb[0] + offset_x, bb[1], bb[2], bb[3]))

    # Debug
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

    # Image enhancement
    if invert:
        (blue_min, blue_max, yellow_min, yellow_max) = (yellow_min, yellow_max, blue_min, blue_max)
        (white_min, white_max, black_min, black_max) = (black_min, black_max, white_min, white_max)
    img = preprocess(img)

    # Find the cones
    init_time = time()
    blue_boxes = find_bouding_boxes(img, blue_min, blue_max, white_min, white_max)
    yellow_boxes = find_bouding_boxes(img, yellow_min, yellow_max, black_min, black_max)
    print("Total time: {}".format(time() - init_time))

    # Plot the results
    if invert:
        img = cv2.bitwise_not(img)

    plot_results(img, blue_boxes, yellow_boxes)

    if save_image:
        img = cv2.bitwise_not(img)
        cv2.imwrite(path+'/output/output.png', generate_output(img, blue_boxes, yellow_boxes))

else:
    # Load the video
    rospack = rospkg.RosPack()
    rospack.list()
    path = rospack.get_path('detection_classic')
    file_name = path + "/images/kth.mp4"
    video = cv2.VideoCapture(file_name)

    # Create video saver
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = cv2.VideoWriter(path+'/output/output.avi', fourcc, 30.0, (1920, 1080))

    if invert:
        (blue_min, blue_max, yellow_min, yellow_max) = (yellow_min, yellow_max, blue_min, blue_max)
        (white_min, white_max, black_min, black_max) = (black_min, black_max, white_min, white_max)

    while video.isOpened():
        ret, frame = video.read()

        frame = preprocess(frame)
        blue_boxes = find_bouding_boxes(frame, blue_min, blue_max, white_min, white_max)
        yellow_boxes = find_bouding_boxes(frame, yellow_min, yellow_max, black_min, black_max)

        output = generate_output(frame, blue_boxes, yellow_boxes)
        if save_video:
            video_out.write(output)
        cv2.imshow('test', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    if save_video:
        video_out.release()
    cv2.destroyAllWindows()
