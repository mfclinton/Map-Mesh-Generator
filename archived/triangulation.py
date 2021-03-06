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
    return (difference[1] + difference[0] <= 1)

def GetLineOfBestFit(indexes_contained_range, points):
    selected_points = points[indexes_contained_range[0]:indexes_contained_range[1] + 1]
    x_and_y_means = np.mean(selected_points, axis=0)
    xy_mean = np.mean(np.multiply(selected_points[:,0], selected_points[:,1]))
    x_sqrd_mean = np.mean(np.square(selected_points[:,1]))
    m_denominator = (math.pow(x_and_y_means[1],2) - x_sqrd_mean)
    if(m_denominator == 0):
        # print("Denominator is 0")
        return None
    m = ((x_and_y_means[1] * x_and_y_means[0]) - xy_mean) / m_denominator
    b = x_and_y_means[0] - m * x_and_y_means[1]
    return (m, b, x_and_y_means[0])

def GetScore(indexes_contained_range, points, m_and_b_y_mean):
    if(m_and_b_y_mean == None):
        return 0 #TODO: need a better solution for undefined

    mean_ae = 0
    max_ae = None
    for index in range(indexes_contained_range[0], indexes_contained_range[1] + 1):
        p = points[index]
        y_approx = m_and_b_y_mean[0] * p[1] + m_and_b_y_mean[1]
        ae = abs(p[0] - y_approx)
        mean_ae += ae
        if(max_ae == None):
            max_ae = ae
        elif(ae > max_ae):
            max_ae = ae

    mean_ae /= (indexes_contained_range[1] + 1) - indexes_contained_range[0]
    # print("S is ", mean_ae)
    return mean_ae

def PlotLines(img, points, lines):
    i = 0
    # print(lines)
    # print(len(lines))
    for l in lines:
        index_1 = l[0][0]
        index_2 = l[0][1]
        if(l[1] == None):
            # print("UHHH")
            cv2.line(img, (points[index_1][1], points[index_1][0]), (points[index_2][1], points[index_2][0]), [0,0,0])
            continue

        line_eq = lambda x: l[1][0] * x + l[1][1]
        x_1 = points[index_1][1]
        x_2 = points[index_2][1]
        # y_1 = int(line_eq(x_1))
        # y_2 = int(line_eq(x_2))
        y_1 = points[index_1][0]
        y_2 = points[index_2][0]
        #TODO, solve the steepness problem
        # if(abs(l[1][0]) > 10):
        #     y_1 = points[index_1][0]
        #     y_2 = points[index_2][0]
            
        cv2.line(img, (x_1, y_1), (x_2, y_2), [0,0,0])
        # cv2.imwrite("line_%d.png"%i, img)
        # print((x_1, y_1), " TO ", (x_2, y_2), " for range of ", l[0], l[1][0])
        i += 1

    # cv2.line(img, (points[-1][1], points[-1][0]), (points[0][1], points[0][0]), [0,0,0])

    
            

def SplitByBestFit(img, edge_dict):
    for color_key in edge_dict.keys():
        
        #debugging purposes
        if(color_key == "255,255,255"):
            continue

        edge_list_num = 0
        for edge_list in edge_dict[color_key]:
            print("{0}_{1}".format(color_key,edge_list_num))
            lines = []
            cur_index = 0
            while(cur_index < len(edge_list) - 1):
                # print(cur_index)

                line_added = False
                for index in range(cur_index + 2, len(edge_list)):
                    points_contained = (cur_index, index)
                    line_info = GetLineOfBestFit(points_contained, edge_list)
                    # r_sqrd = GetRSqred(points_contained, edge_list, line_info)
                    score = GetScore(points_contained, edge_list, line_info)
                    if(2*score > 1):
                        lines.append(((cur_index, index - 1), line_info[:2]))
                        # print(index - cur_index, "index now is ", index - 1)
                        # print("INDEX CHANGE ", cur_index, "TO ", index - 1)
                        cur_index = index - 1
                        line_added = True
                        break

                if(not line_added):
                    # print("Should Be At End")
                    points_contained = (cur_index, len(edge_list) - 1)
                    # print("SPECIAL CONTAINED " , points_contained)
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
            # PlotLines(img, edge_list, lines)
            print("before triangulate")
            Triangulate(edge_list, lines, "{0}_{1}".format(color_key,edge_list_num))
            edge_list_num += 1

def ConvertDictToNPArrays(edge_dict):
    for color_key in edge_dict.keys():
        for index in range(len(edge_dict[color_key])):
            edge_dict[color_key][index] = np.array(edge_dict[color_key][index])

def Triangulate(points, lines, objName):
    if(len(lines) < 3):
        print("too small")
        return

    segments = []
    vertices = []
    i = 0
    for l in lines:
        # point to point, temp TODO: change tis
        index_1 = l[0][0]
        vertices.append((points[index_1][0], points[index_1][1]))
        segments.append((i, i+1))
        i += 1

    last_point = points[lines[-1][0][0]]
    vertices.append((last_point[0],last_point[1]))

    
    
    # TODO: temp fix to connect last part
    segments.append((len(segments), 0))

    # print(vertices)
    # print(segments)
    print("before B")
    try:
        A = dict(vertices=vertices, segments=segments)
        print("1")
        B = tr.triangulate(A,"p")
        # print(B)
        TriangulationToOBJ(B, objName)
    except:
        print("Exception Occured")
    
    # tr.compare(plt, A, B)
    # plt.show()
    print("UH")
    # i = 0
    # print(lines)
    # # print(len(lines))
    # for l in lines:
    #     index_1 = l[0][0]
    #     index_2 = l[0][1]
    #     if(l[1] == None):
    #         cv2.line(img, (points[index_1][1], points[index_1][0]), (points[index_2][1], points[index_2][0]), [0,0,0])
    #         continue

    #     line_eq = lambda x: l[1][0] * x + l[1][1]
    #     x_1 = points[index_1][1]
    #     x_2 = points[index_2][1]

    #     y_1 = points[index_1][0]
    #     y_2 = points[index_2][0]
            
    #     cv2.line(img, (x_1, y_1), (x_2, y_2), [0,0,0])
    #     i += 1

def TriangulationToOBJ(B, objName):
    print("wowza oh nooo")
    if("vertices" not in B or "triangles" not in B):
        return

    with open("objs\\{0}.obj".format(objName),"w") as f:
        for v in B["vertices"]:
            f.write("v {0} {1} {2}\n".format(v[0],0,v[1]))

        # TODO: temp for flat
        f.write("vn {0} {1} {2}\n".format(0,1,0))

        for t in B["triangles"]:
            f.write("f {0}//1 {1}//1 {2}//1\n".format(t[0] + 1,t[1] + 1,t[2] + 1))

#Parameters
fileName = "test_files/North_South_America.png"

#Image Processing
img = cv2.imread(fileName) #(row, column) order

edge_dict = {}
for index in np.ndindex(img.shape[:-1]):
    if(IsEdge(index, img)):
        color_key = ",".join(["%d"%value for value in img[index]])

        #dict entry creation
        #TODO more efficient
        if(not color_key in edge_dict):
            edge_dict[color_key] = []

        #adding to edge list
        edge_list_indexes_to_merge = []
        for edge_list_index, edge_list in enumerate(edge_dict[color_key]):
            if(IsNeighbor(index, edge_list[-1])):
                edge_list.append(index)
                edge_list_indexes_to_merge.append(edge_list_index)
            elif(IsNeighbor(index, edge_list[0])):
                edge_list.insert(0, index)
                edge_list_indexes_to_merge.append(edge_list_index)

        if(len(edge_list_indexes_to_merge) == 0):
            edge_dict[color_key].append([index])
        elif(len(edge_list_indexes_to_merge) == 2):
            merged_edge_list = edge_dict[color_key][edge_list_indexes_to_merge[0]]
            if(merged_edge_list[0] == index):
                merged_edge_list.reverse() #flips list to be [..., index]

            other_edge_list = edge_dict[color_key][edge_list_indexes_to_merge[1]]
            if(other_edge_list[-1] == index):
                other_edge_list.reverse() #flips list to be [index, ...]
            other_edge_list.pop(0)

            merged_edge_list.extend(other_edge_list)
            edge_dict[color_key].pop(edge_list_indexes_to_merge[1])

        elif(len(edge_list_indexes_to_merge) > 2):
            print("Image Is Not Properly Formatted")
            break

#splitting into line segments
# new_img = np.full(img.shape,[255,255,255], np.uint8)
ConvertDictToNPArrays(edge_dict)
SplitByBestFit(img, edge_dict)
            

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
print("fin")
# plt.show()
# cv2.imshow("image", img)
# cv2.waitKey(0)
