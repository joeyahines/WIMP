# import the necessary packages
import numpy as np
import cv2

errplus = lambda t: t + (t * 0.05)
errorplusvev = np.vectorize(errplus)
errminus = lambda t: t - (t * 0.05)
errorminusvev = np.vectorize(errminus)

# load the image
org_image = cv2.imread("test2.png")

# define the list of boundaries
colors = [
    [65, 200, 230],
    [95, 200, 240],
    [230, 229, 236],
    [70, 190, 240],
]
low = np.array([225, 225, 225], dtype="uint8")
up = np.array([255, 255, 255], dtype="uint8")
mask = cv2.inRange(org_image, low, up)
image = cv2.bitwise_and(org_image, org_image, mask=255 - mask)

font = cv2.FONT_HERSHEY_SIMPLEX

# loop over the boundaries
for color in colors:
    # create NumPy arrays from the boundaries
    arr = np.array(color, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask

    mask = cv2.inRange(image, errorminusvev(arr), errorplusvev(arr))
    output = cv2.bitwise_and(image, image, mask=mask)

    height, width, channels = output.shape
    cv2.putText(output, color.__str__(), (width - 150, height - 20), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # show the images
    cv2.imshow("images", np.hstack([org_image, image, output]))
    cv2.waitKey(0)
