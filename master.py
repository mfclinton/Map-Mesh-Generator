import numpy as np
import cv2
import matplotlib.pyplot as plt
from lib.worker import *

# Extracts coordinates for a bounding box around each respective color
def CreateBoundingBoxes(img):
    b_boxes = {}
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            key = img[y,x].tobytes()
            b_box = b_boxes.setdefault(key, ([y,x],[y,x]))

            # Finds minimum and maximum coordinates to bounding box each color
            if(y < b_box[0][0]):
                b_box[0][0] = y
            if(x < b_box[0][1]):
                b_box[0][1] = x
            if(y > b_box[1][0]):
                b_box[1][0] = y
            if(x > b_box[1][1]):
                b_box[1][1] = x
    
    return b_boxes

# Displays the bounding box region
def DebugBoundingBox(img, b_boxes, key):
    b_box = b_boxes[key]
    y1, x1 = b_box[0]
    y2, x2 = b_box[1]
    cv2.imshow("bounding box key: " + str(key), img[y1:y2+1,x1:x2+1])
    cv2.waitKey(0)

def ThreadedProcessImage(img, b_boxes):
    threads = []
    # Start each thread
    for key in b_boxes.keys():
        color = np.frombuffer(key, dtype=img.dtype)
        worker_thread = Worker(b_boxes[key], color)
        worker_thread.start()
        threads.append(worker_thread)

    # Wait for threads to finish
    for t in threads:
        t.join()


img_path = "test_files\\North_South_America.png"
img = cv2.imread(img_path)
b_boxes = CreateBoundingBoxes(img)
# DebugBoundingBox(img, b_boxes, list(b_boxes.keys())[1])
ThreadedProcessImage(img, b_boxes)

