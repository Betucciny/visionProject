import cv2
import math
from functions import *
import numpy as np
import time

tag_ids = ['0101', '0111', '1111']
img_paths = ['1.jpg', '2.jpg', '3.jpg']

# read the images into memory
imgs = []
for path in img_paths:
    imgs.append(cv2.imread(path))


# Setup camera
cap = cv2.VideoCapture(0)

# While loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # find correct contours
    [all_cnts, cnts] = findcontours(frame, 180)
    # approximate quadralateral to each contour and extract corners
    [tag_cnts, corners] = approx_quad(cnts)
    # cv2.drawContours(frame, all_cnts, -1, (0, 255, 0), 4)
    cv2.drawContours(frame, tag_cnts, -1, (255, 0, 0), 4)

    for i, tag in enumerate(corners):
        # find number of points in the polygon
        num_points = num_points_in_poly(frame, tag_cnts[i])

        # set the dimension for homography
        dim = int(math.sqrt(num_points))

        # compute homography, for the forward warp we need the inverse
        H = homography(tag, dim)
        H_inv = np.linalg.inv(H)

        # get squared tag
        square_img = warp(H_inv, frame, dim, dim)

        # threshold the squared tag
        imgray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
        ret, square_img = cv2.threshold(imgray, 180, 255, cv2.THRESH_BINARY)

        # decode squared tile
        [tag_img, id_str] = encode_tag(square_img)

        if id_str in tag_ids:
            index = tag_ids.index(id_str)
            new_img = imgs[index]
        else:
            continue

        # superimpose the image on the tag
        dim = new_img.shape[0]
        H = homography(tag, dim)
        h = frame.shape[0]
        w = frame.shape[1]
        frame1 = warp(H, new_img, h, w)
        frame2 = blank_region(frame, tag_cnts[i], 0)
        frame = cv2.bitwise_or(frame1, frame2)

    cv2.imshow('WebCam', frame)

    # wait for the key and come out of the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Discussed below
cap.release()
cv2.destroyAllWindows()