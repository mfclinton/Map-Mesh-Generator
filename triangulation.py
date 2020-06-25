import numpy as np
import matplotlib.pylab as plt
import cv2
import triangle as tr
import time
import math

#Helper Functions

#Checks if pixel at index is an edge
offsets = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
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
    difference = np.abs(np.subtract(index_a,index_b))
    # if(difference[1] + difference[0] <= 1):
    #     print(difference)
    return (difference[1] + difference[0] <= 1)
    # return (difference[1] <= 1 and difference[0] <= 1)

# TODO: Problem, need percentage
# def LineErrorPtoP(line_segment, points):
#     p1 = points[line_segment[0]]
#     p2 = points[line_segment[1]]

#     if(p2[1] - p1[1] == 0):
#         error = 0
#         for index in range(line_segment[0] + 1,line_segment[1]):
#             error += math.sqrt(math.pow(points[index][1] - p1[1],2))
#         return error

#     m = (p2[0] - p1[0]) / (p2[1] - p1[1]) #y axis, gonna be upsidedown
#     b = p1[0] - m*p1[1]
#     proj_denominator = 1 + (m*m)
#     error = 0
#     for index in range(line_segment[0] + 1,line_segment[1]):
#         p = points[index]
#         proj_scalar = (p[1] + p[0] * m) / proj_denominator
#         proj = (proj_scalar*m, proj_scalar) #row, column
#         dist = math.sqrt(math.pow(proj[0] - p[0],2) + math.pow(proj[1] - p[1],2))
#         error += dist
    
#     return error

def GetLineOfBestFit(indexes_contained_range, points):
    selected_points = points[indexes_contained_range[0]:indexes_contained_range[1] + 1]
    # print(selected_points)
    x_and_y_means = np.mean(selected_points, axis=0)
    xy_mean = np.mean(np.multiply(selected_points[:,0], selected_points[:,1]))
    x_sqrd_mean = np.mean(np.square(selected_points[:,0]))

    m_denominator = (math.pow(x_and_y_means[1],2) - x_sqrd_mean)
    if(m_denominator == 0):
        print("Denominator is 0")
        return None
    m = (x_and_y_means[1] * x_and_y_means[0] - xy_mean) / m_denominator
    b = x_and_y_means[0] - m * x_and_y_means[1]
    return (m, b, x_and_y_means[0])

def GetRSqred(indexes_contained_range, points, m_and_b_y_mean):
    if(m_and_b_y_mean == None):
        return 1 #TODO: need a better solution for undefined

    total_sqred_error_line = 0
    total_sqred_error_mean = 0
    for index in range(indexes_contained_range[0], indexes_contained_range[1] + 1):
        p = points[index]
        y_approx = m_and_b_y_mean[0] * p[1] + m_and_b_y_mean[1]

        total_sqred_error_line += math.pow(p[0] - y_approx, 2)
        total_sqred_error_mean += math.pow(p[0] - m_and_b_y_mean[2], 2)

    if(total_sqred_error_mean == 0):
        # print(total_sqred_error_line)
        return 1 #TODO: check this

    return 1 - (total_sqred_error_line / total_sqred_error_mean)

def PlotLines(img, points, lines):
    for l in lines:
        index_1 = l[0][0]
        index_2 = l[0][1]
        if(l[1] == None):
            cv2.line(img, (points[index_1][1], points[index_1][0]), (points[index_2][1], points[index_2][0]), [0,0,0])
            continue

        line_eq = lambda x: l[1][0] * x + l[1][1]
        x_1 = points[index_1][1]
        x_2 = points[index_2][1]
        y_1 = int(line_eq(x_1))
        y_2 = int(line_eq(x_2))
        cv2.line(img, (x_1, y_1), (x_2, y_2), [0,0,0])

    # cv2.line(img, (points[-1][1], points[-1][0]), (points[0][1], points[0][0]), [0,0,0])

    
            

def SplitByBestFit(img, edge_dict):
    for color_key in edge_dict.keys():
        #debugging purposes
        if(color_key == "255,255,255"):
            continue
        # if(color_key != "36,28,237"):
        #     continue

        # print(color_key)
        # print(len(edge_dict[color_key]))
        for edge_list in edge_dict[color_key]:
            lines = []
            cur_index = 0
            while(cur_index < len(edge_list) - 1):
                # print(cur_index)

                line_added = False
                for index in range(cur_index + 2, len(edge_list)):
                    points_contained = (cur_index, index)
                    line_info = GetLineOfBestFit(points_contained, edge_list)
                    r_sqrd = GetRSqred(points_contained, edge_list, line_info)
                    if(r_sqrd < .7):
                        lines.append(((cur_index, index - 1), line_info[:2]))
                        cur_index = index - 1
                        line_added = True
                        # print(r_sqrd)
                        break

                if(not line_added):
                    points_contained = (cur_index, len(edge_list) - 1)
                    line_info = GetLineOfBestFit(points_contained, edge_list)
                    if(line_info == None):
                        lines.append(((cur_index, len(edge_list) - 1), None))
                        break
                    lines.append(((cur_index, len(edge_list) - 1), line_info[:2]))
                    break

                #pair cur_index with cur_index + 1
            
            # print(lines)
            # print(color_key, " : ", len(lines), " vs ", len(edge_list))
            # if(len(edge_list) == 1):
            #     print("error", color_key, edge_list)
            PlotLines(img, edge_list, lines)

def ConvertDictToNPArrays(edge_dict):
    for color_key in edge_dict.keys():
        for index in range(len(edge_dict[color_key])):
            edge_dict[color_key][index] = np.array(edge_dict[color_key][index])


#Parameters
fileName = "test_files/europe.png"

#Image Processing
img = cv2.imread(fileName) #(row, column) order

edge_dict = {}
for index in np.ndindex(img.shape[:-1]):
    if(IsEdge(index, img)):
        color_key = ",".join(["%d"%value for value in img[index]])
        # if(not color_key == "29,230,181"): # REMOVE LATER
        #     continue
        # print(index)
        #dict entry creation
        #TODO more efficient
        if(not color_key in edge_dict):
            edge_dict[color_key] = []

        #adding to edge list
        edge_list_indexes_to_merge = []
        for edge_list_index, edge_list in enumerate(edge_dict[color_key]):
            # print(edge_list[0], edge_list[-1])
            if(IsNeighbor(index, edge_list[-1])):
                edge_list.append(index)
                edge_list_indexes_to_merge.append(edge_list_index)
            elif(IsNeighbor(index, edge_list[0])):
                edge_list.insert(0, index)
                edge_list_indexes_to_merge.append(edge_list_index)

        if(len(edge_list_indexes_to_merge) == 0):
            edge_dict[color_key].append([index])
        elif(len(edge_list_indexes_to_merge) == 2):
            print("MERGE", len(edge_dict[color_key]))
            # if(color_key == "36,28,237"):
            #     print(index)
            merged_edge_list = edge_dict[color_key][edge_list_indexes_to_merge[0]]
            if(merged_edge_list[0] == index):
                merged_edge_list.reverse() #flips list to be [..., index]

            other_edge_list = edge_dict[color_key][edge_list_indexes_to_merge[1]]
            if(other_edge_list[-1] == index):
                other_edge_list.reverse() #flips list to be [index, ...]
            other_edge_list.pop(0)

            merged_edge_list.extend(other_edge_list)
            edge_dict[color_key].pop(edge_list_indexes_to_merge[1])
            # print(edge_dict[color_key])
            print("done", len(edge_dict[color_key]))

        elif(len(edge_list_indexes_to_merge) > 2):
            print("Image Is Not Properly Formatted")
            break

#splitting into line segments
new_img = np.full(img.shape,[255,255,255], np.uint8)
ConvertDictToNPArrays(edge_dict)
SplitByBestFit(new_img, edge_dict)

# # This is for point-to-point lines
# for color_key in edge_dict.keys():
#     if(color_key == "255,255,255"):
#         continue
#     if(color_key != "36,28,237"):
#         continue

#     for edge_list in edge_dict[color_key]:
#         line_segments = []
        
#         cur_index = 0
#         print(len(edge_list))
#         while(cur_index < len(edge_list) - 1):
#             cur_index_updated = False
#             for edge_index, edge in enumerate(edge_list):
#                 if(edge_index == cur_index):
#                     # print("LOL")
#                     continue
#                 error = LineError((cur_index, edge_index), edge_list)
#                 if(error > 20):
#                     print(cur_index, edge_index)
#                     cur_index = edge_index
#                     line_segments.append((cur_index, edge_index - 1))
#                     cur_index_updated = True
#                     break
            
#             if(not cur_index_updated):
#                 print("special case", cur_index, edge_index)
#                 cur_index += 2
#                 # line_segments.append((cur_index, cur_index + 1))
        
#         #debugging
#         for line in line_segments:
#             print(line)
#             p1 = edge_list[line[0]]
#             p2 = edge_list[line[1]]
#             cv2.line(img, (p1[1], p1[0]), (p2[1], p2[0]), [0,0,0])



            

#debugging
# frame = 0
# for color_key in edge_dict.keys():
#     if(color_key == "255,255,255"):
#         continue
#     # if(len(edge_dict[color_key]) < 3):
#     #     continue
#     # print(color_key)
#     # print(len(edge_dict[color_key]))
#     j = -1
#     for edge_list in edge_dict[color_key]:
#         j += 1
#         # if(not j == 1):
#         #     continue
#         # print(len(edge_list))
#         i = 0
#         # if(j == 1):
#         #     continue
#         # j += 1
#         # print(edge_list)
#         # print(len(edge_dict[color_key]))
#         for edge in edge_list:
#             img[edge] = [0,0,0]
#             i += 1
#             frame += 1
        
    

cv2.imwrite("output.png", new_img)
# cv2.imshow("image", img)
# cv2.waitKey(0)
