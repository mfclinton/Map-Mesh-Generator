import numpy as np
import matplotlib.pylab as plt
import cv2
import triangle as tr
import time

#Helper Functions

#Checks if pixel at index is an edge
offsets = [(-1,0),(1,0),(0,-1),(0,1)]
def IsEdge(index, img):
    color = img[index]
    for o in offsets:
        neighbor_index = (index[0] + o[0], index[1] + o[1])
        if(not (0 <= neighbor_index[0] < img.shape[0] and 0 <= neighbor_index[1] < img.shape[1])):
            continue

        if(not np.all(img[neighbor_index] == color)):
            return True
    return False

def IsNeighbor(index_a, index_b):
    total = np.sum(np.abs(np.subtract(index_a,index_b)))
    return total <= 2

#Parameters
fileName = "test_files/europe.png"

#Image Processing
img = cv2.imread(fileName) #(row, column) order

edge_dict = {}
for index in np.ndindex(img.shape[:-1]):
    if(IsEdge(index, img)):
        color_key = ",".join(["%d"%value for value in img[index]])

        #dict entry creation
        if(not color_key in edge_dict):
            edge_dict[color_key] = [[index]]
            continue

        #adding to edge list
        edge_list_indexes_to_merge = []
        for edge_list_index, edge_list in enumerate(edge_dict[color_key]):
            if(IsNeighbor(index, edge_list[-1])):
                edge_list.append(index)
                edge_list_indexes_to_merge.append(edge_list_index)
            elif(IsNeighbor(index, edge_list[0])):
                edge_list.insert(0, index)
                edge_list_indexes_to_merge.append(edge_list_index)

        #merge list
        # if(len(edge_list) > 1):
        #     print(color_key) #TODO

#debugging
for color_key in edge_dict.keys():
    if(color_key == "255,255,255"):
        continue
    print(color_key)
    for edge_list in edge_dict[color_key]:
        i = 0
        for edge in edge_list:
            img[edge] = [i,i,i]
            i += 5
    

cv2.imwrite("output.png", img)
# cv2.imshow("image", img)
# cv2.waitKey(0)
