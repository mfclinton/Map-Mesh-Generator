import cv2
import numpy as np
import math

#Fetches a random pixel
def GetRandomPoint(roi):
    possible_starts = np.where(roi != 0)
    rng = np.random.randint(0,len(possible_starts[0]))
    return np.array([possible_starts[0][rng],possible_starts[1][rng]])

#picks starting pixel from several options
def GetStartingPoint(roi, option=None):
    origin = GetRandomPoint(roi)
    return np.array([8,37])
    return origin

def CreateNewPoint(origin, dir_vec, roi):
    #Unit Direction Vector
    dir_vec /= np.linalg.norm(dir_vec, ord=1)
    
    cur_pixel = origin
    position = origin.astype(float)

    while(True):
        position += dir_vec
        next_pixel = np.rint(position).astype(int)
        
        #Check if in bounds, then if there is a change in color values
        if(not (0 <= next_pixel[0] < roi.shape[0] and 0 <= next_pixel[1] < roi.shape[1])):
            break
        elif(roi[cur_pixel[0], cur_pixel[1]] != roi[next_pixel[0], next_pixel[1]]):
            cur_pixel = cur_pixel if roi[next_pixel[0], next_pixel[1]] == 0 else next_pixel
            break

        cur_pixel = next_pixel
    return cur_pixel

def IsInsidePolygon(pos, points):
    num_intersects = 0
    print("IS INSIDE POLYGON------")
    for p_idx in range(-1, len(points) - 1):
        p1 = points[p_idx][0]
        p2 = points[p_idx + 1][0]

        #Checks that pos is to between p1 and p2 on the y axis, and left to at least one of the points
        if(not ((p1[0] <= pos[0] <= p2[0] or p2[0] <= pos[0] <= p1[0]) and (pos[1] <= p1[1] or pos[1] <= p2[1]))):
            continue

        #Checks if new point is equal to an existing point
        if(np.all(pos == p1)):
            #TODO, don't add new point
            return True
        
        #Checks Vertical and Horizontal Lines
        if(p1[0] == p2[0]):
            if(p1[1] <= pos[1] <= p2[1] or p2[1] <= pos[1] <= p1[1]):
                return True #On the line
            continue
        elif(p1[1] == p2[1]):
            if(pos[1] == p1[1]):
                return True #On the line
            num_intersects += 1
            continue

        m = (p1[0] - p2[0]) / (p1[1] - p2[1])
        b = p1[0] - m * p1[1]
        x_calculated = (pos[0] - b) / m
        #Return true if on the line
        if(pos[1] == x_calculated):
            return True
        elif(pos[1] < x_calculated):
            if(x_calculated == p1[1]):
                p_prev = points[p_idx - 1][0]
                if(p_prev[0] <= pos[0] <= p2[0] or p2[0] <= pos[0] <= p_prev[0]):
                    num_intersects += 1
                continue
            elif(x_calculated == p2[1]):
                continue


            print(p1, p2, "INTERSECT")
            num_intersects += 1
    print(num_intersects)
    print("ISOUTISE------")
    return (num_intersects % 2) == 1

#creates the initial triangle and returns a list of 3 points and their respective ids
def CreateStartingTriangle(roi):
    origin = GetStartingPoint(roi)
    initial_dirs = [np.array([-1.0,0.0]), np.array([1/2, math.sqrt(3)/2]), np.array([1/2, -math.sqrt(3)/2])]
    points = []

    for pid, dir_vector in enumerate(initial_dirs):
        points.append((CreateNewPoint(origin, dir_vector, roi), pid + 1))

    #Returns points, and triangle list
    return points, [(1,2,3)]

def GetNewOriginAndDir(points, p_idx1, p_idx2, roi):
    p1 = points[p_idx1][0]
    p2 = points[p_idx2][0]
    new_origin_pos = (p1 + p2) / 2 #TODO: issue with position and rounding, fix later
    new_origin_pixel = np.rint(new_origin_pos).astype(int)

    #Get normal vector, either calculated or vertical by default (because divide by zero)
    dir_vec = np.array([1.0,0.0])
    if(p1[0] != p2[0]):
        dir_vec =  np.array([- (p2[1] - p1[1]) / (p2[0] - p1[0]), 1])
        dir_vec /= np.linalg.norm(dir_vec, ord=1)

    if(IsInsidePolygon(new_origin_pos + dir_vec, points) == IsInsidePolygon(new_origin_pos - dir_vec, points)):
        print("----------")
        print("WTFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
        print(points, new_origin_pos + dir_vec)
        print(p1, p2)
        print(new_origin_pos, dir_vec)
        for p in points:
            roi[p[0][0],p[0][1]] = 180

        meme1 = np.rint(new_origin_pos + dir_vec).astype(int)
        meme2 = np.rint(new_origin_pos - dir_vec).astype(int)
        print(meme1, meme2)
        roi[meme1[0], meme1[1]] = 127
        roi[meme2[0], meme2[1]] = 127
        roi[new_origin_pixel[0], new_origin_pixel[1]] = 50
        # cv2.imwrite("debug_pic.png", roi)
        # cv2.imshow("test", roi)
        # cv2.waitKey(0)

        print("----------")
        

    #Sets dir_vec to point outwards from polygon
    if(IsInsidePolygon(new_origin_pos + dir_vec, points)):
        dir_vec *= -1
        print("BBB")
    else:
        print("AAA")

    #Sets dir_vec to point inwards if the new_origin is not in the actual shape
    if(roi[new_origin_pixel[0], new_origin_pixel[1]] == 0):
        dir_vec *= -1
        #NOT THE ISSUE
        # print(dir_vec, new_origin, len(points))

    return new_origin_pixel, dir_vec


def ProcessRegion(roi, num_divides, output_name):
    points, triangles = CreateStartingTriangle(roi)
    new_id = 4 #Point ID
    for div_idx in range(num_divides):
        for p_idx in range(-1, (len(points) * 2) - 1, 2):
            new_origin, dir_vec = GetNewOriginAndDir(points, p_idx, p_idx + 1, roi)
            new_point = CreateNewPoint(new_origin, dir_vec, roi)

            triangles.append((points[p_idx][1], points[p_idx + 1][1], new_id))
            points.insert(p_idx + 1, (new_point, new_id))
            new_id += 1
    points.sort(key = lambda x: x[1])
    WriteToOutput(output_name, points, triangles)

def WriteToOutput(file_name, points, triangles):
    with open(file_name, "w") as f:
        for p in points:
            line = "v " + str(p[0][1]) + " " + str(p[0][0]) + " " + str("0") + "\n"
            f.write(line)
        for t in triangles:
            line = "f " + str(t[0]) + " " + str(t[1]) + " " + str(t[2]) + "\n"
            f.write(line)
            

def Main(img_path, num_divides, ignored_cells, min_bb_area):
    #Image variables
    img = cv2.imread(img_path)
    h,w, channels = img.shape

    #Flood fill helper variables
    floodflags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    flood_value = 255

    #Mask used to keep track of explored areas
    explored_mask = np.zeros((h + 2, w + 2), np.uint8)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(np.any(np.all(img[y,x] == ignored_cells, axis=1))):
                continue
            elif(explored_mask[y+1,x+1] != 0):
                #Mask pixels are off by 1, we skip areas already filled in (IE: non-zero)
                continue

            #Makes a new mask (region of interest) to fill with the new shape, adds area to explored mask
            roi = np.zeros((h + 2, w + 2), np.uint8)
            retval, _, _, bounding_box = cv2.floodFill(img, roi, (x,y), 0, 5, 5, floodflags) #TODO check 5 value
            explored_mask = cv2.bitwise_or(explored_mask, roi)

            if(bounding_box[2] * bounding_box[3] < min_bb_area):
                #Area is too small, so we skip
                continue

            #Crops to region filled it, speeds up future calculations
            roi = roi[bounding_box[1]+1:bounding_box[1] + bounding_box[3] + 1, bounding_box[0] + 1:bounding_box[0] + bounding_box[2] + 1]
            output_name = str(img[y,x][2]) + "_" + str(img[y,x][1]) + "_" + str(img[y,x][0]) + ".obj"
            ProcessRegion(roi, num_divides, output_name)

            

if __name__ == "__main__":
    img_path = "test_files\\red_ring.png"

    #Options for flood fill
    num_divides = 10
    ignored_cells = [[255,255,255]]
    min_bb_area = 25

    Main(img_path, num_divides, ignored_cells, min_bb_area)








