# import the necessary packages
import cv2
import imutils
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
origin_image = cv2.imread("test2.png")

low = np.array([190, 190, 190], dtype="uint8")
up = np.array([255, 255, 255], dtype="uint8")
mask = cv2.inRange(origin_image, low, up)
image = cv2.bitwise_and(origin_image, origin_image, mask=255 - mask)

resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY + 8)[1]

# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# loop over the contours
for c in cnts:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)

    color = image[cY, cX, 0:]
    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(origin_image, [c], -1, (0, 255, 0), 2)
    cv2.putText(origin_image, str(color), (cY, cX), font, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
    # show the output image
    cv2.imshow("Image", origin_image)

cv2.waitKey(0)
