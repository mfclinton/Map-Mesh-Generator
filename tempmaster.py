import cv2
import numpy as np
import math

def Rotate(dir_vec, radians):
    new_y = -dir_vec[1] * math.sin(radians) + dir_vec[0] * math.cos(radians)
    new_x = dir_vec[1] * math.cos(radians) + dir_vec[0] * math.sin(radians)
    magnitude = math.sqrt(new_y**2 + new_x**2)
    return new_y / magnitude, new_x / magnitude

def CreateNewPoint(origin, dir_vec, roi, cur_id):
    cur_pixel = None
    cur_position = origin.astype(float)
    # print(mask[int(cur_pixel[0] + dir_vec[0]), int(cur_pixel[1] + dir_vec[1])])
    while(True):
        prev_pixel = cur_pixel
        cur_pixel = np.rint(cur_position).astype(int)
        # print(roi[cur_pixel[0], cur_pixel[1]])
        if((0 <= cur_pixel[0] < roi.shape[0]) and (0 <= cur_pixel[1] < roi.shape[1]) and np.all(roi[cur_pixel[0], cur_pixel[1]] != 0)):
            roi[cur_pixel[0], cur_pixel[1]] = 1
            cur_position += dir_vec
        else:
            cur_pixel = prev_pixel
            break
    if(cur_pixel is None):
        return (origin, cur_id) #TODO: verify
    return (cur_pixel, cur_id)

def GetRandomPoint(roi):
    possible_starts = np.where(roi != 0)
    rng = np.random.randint(0,len(possible_starts[0]))
    origin = np.array([possible_starts[0][rng],possible_starts[1][rng]])
    return origin

def CreateStartingTriangle(roi):
    #Picks Random Starting Pixel
    origin = GetRandomPoint(roi)
    dir_vec = np.array([-1.0,0.0])

    points = []
    points.append(CreateNewPoint(origin.astype(float), dir_vec, roi, 1))
    dir_vec = Rotate(dir_vec, 2 * math.pi / 3)
    points.append(CreateNewPoint(origin.astype(float), dir_vec, roi, 2))
    dir_vec = Rotate(dir_vec, 2 * math.pi / 3)
    points.append(CreateNewPoint(origin.astype(float), dir_vec, roi, 3))
    return points

#Checks if a pixel "pos" is inside the polygon using a simple line intersection algorithm
def IsInsidePolygon(pos, points):
    intersects = 0
    for p_idx in range(-1, len(points) - 1):
        p1 = points[p_idx][0]
        p2 = points[p_idx + 1][0]

        if(((p1[0] <= pos[0] <= p2[0]) or (p2[0] <= pos[0] <= p1[0])) and ((pos[1] <= p1[1]) or (pos[1] <= p2[1]))):
            if(p1[1] == p2[1] or p1[0] == p2[0]):
                intersects += 1
                continue

            # print(p1, p2)
            m = (p1[0] - p2[0]) / (p1[1] - p2[1])
            b = p1[0] - m * p1[1]
            x_calculated = (pos[0] - b) / m
            #TODO: Fix issue with wrong slopes
            if(x_calculated >= pos[1]):
                intersects += 1
                # print("SUCCESS")
        # print(p1,p2)
    # print(intersects)
    return (intersects % 2) == 1
        

img_path = "test_files\\red_MA.png"
img = cv2.imread(img_path)
h,w, channels = img.shape

# OPTIONS FOR FLOOD FILL
ignored_cells = [[255,255,255]]
min_area = 25
seed = (50,50)
floodflags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
flood_value = 255
num_divides = 5

explored_mask = np.zeros((h + 2, w + 2), np.uint8)
#Note: rect in format of (x, y, width, height)
for x in range(img.shape[1]):
    for y in range(img.shape[0]):
        if(np.any(np.all(img[y,x] == ignored_cells, axis=1))):
            continue

        #mask offset by 1 pixel
        if(explored_mask[y+1,x+1] == 0):
            mask = np.zeros((h + 2, w + 2), np.uint8)
            retval, _, _, rect = cv2.floodFill(img, mask, (x,y), 0, 5, 5, floodflags) #TODO check 5 value
            explored_mask = cv2.bitwise_or(explored_mask, mask)
            if(rect[2] * rect[3] < min_area):
                print("Too Small Area Size")
                continue
            # roi = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            roi = mask[rect[1]+1:rect[1] + rect[3] + 1, rect[0] + 1:rect[0] + rect[2] + 1]
            points = CreateStartingTriangle(roi)
            triangles = [(1,2,3)]
            cur_id = 4
            for div_idx in range(num_divides):
                for p_idx in range(-1, (len(points) * 2) - 1, 2):
                    p1 = points[p_idx][0]
                    p2 = points[p_idx + 1][0]
                    new_origin = np.rint((p1 + p2) / 2).astype(int)

                    dir_vec = np.array([1,0]) #TODO: Check for verical line
                    if(p1[0] != p2[0]):
                        dir_vec =  np.array([- (p2[1] - p1[1]) / (p2[0] - p1[0]), 1])
                        dir_vec = dir_vec / np.linalg.norm(dir_vec, ord=1)
                    
                    #TODO: add new point
                    if(IsInsidePolygon(new_origin + dir_vec, points)):
                        dir_vec *= -1 #go outwards

                    if(roi[new_origin[0], new_origin[1]] == 0):
                        dir_vec *= -1 #go inwards

                    new_point = CreateNewPoint(new_origin, dir_vec, roi, cur_id)
                    triangles.append((points[p_idx][1], points[p_idx + 1][1], cur_id))
                    cur_id += 1
                    points.insert(p_idx + 1, new_point)

            points.sort(key = lambda x: x[1])
            #ALGORITHM OPTIONS
            with open("debug.obj", "w") as f:
                for p in points:
                    line = "v " + str(p[0][1]) + " " + str(p[0][0]) + " " + str("0") + "\n"
                    f.write(line)
                for t in triangles:
                    line = "f " + str(t[0]) + " " + str(t[1]) + " " + str(t[2]) + "\n"
                    f.write(line) 
                
            cv2.imwrite("debug_pic.png", roi)
            # cv2.imshow("test", roi)
            # cv2.waitKey(0)





    

