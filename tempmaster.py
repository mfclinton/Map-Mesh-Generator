import cv2
import numpy as np

img_path = "test_files\\red_circle.png"
img = cv2.imread(img_path)

h,w, channels = img.shape
mask = np.zeros((h + 2, w + 2), np.uint8)

# OPTIONS FOR FLOOD FILL
seed = (50,50)
floodflags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
flood_value = 255

#Note: rect in format of (x, y, width, height)
retval, _, _, rect = cv2.floodFill(img, mask, seed, 0, 5, 5, floodflags) #TODO check 5 value
roi = mask[rect[1] + 1:rect[1] + rect[3] + 1, rect[0] + 1:rect[0] + rect[2] + 1]
cv2.imshow("test", roi)
cv2.waitKey(0)


