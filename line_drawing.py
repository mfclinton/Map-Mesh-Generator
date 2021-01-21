import cv2
import numpy as np
import math

# def DrawLine(p1, p2, img):
#     #TODO, mark given
#     y_delta = (p2[0] - p1[0])
#     x_delta = (p2[1] - p1[1])

#     m = 2 * y_delta
#     slope_sign = int(m / abs(m))
#     slope_error = m - x_delta

#     y = p1[0]
#     for x in range(p1[1], p2[1] + 1):
#         print(x, y)
#         img[x,y] = (255,255,255)
#         slope_error += m
#         if(slope_error >= 0):
#             y += slope_sign
#             slope_error -= 2 * x_delta
#     cv2.imshow("test", img)
#     cv2.waitKey(0)

def DrawLine(p1, p2, img):


img = np.zeros((50,50,3), np.uint8)
p1 = (2,3)
p2 = (5,15)
DrawLine(p1, p2, img)