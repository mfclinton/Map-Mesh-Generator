import cv2
import numpy as np
import math

def Rotate(dir_vec, radians):
    new_y = -dir_vec[1] * math.sin(radians) + dir_vec[0] * math.cos(radians)
    new_x = dir_vec[1] * math.cos(radians) + dir_vec[0] * math.sin(radians)
    return new_y, new_x

def CreateNewPoint(origin, dir_vec, mask):
    cur_pixel = origin
    print(mask[int(cur_pixel[0] + dir_vec[0]), int(cur_pixel[1] + dir_vec[1])])
    while(np.all(mask[int(cur_pixel[0] + dir_vec[0]), int(cur_pixel[1] + dir_vec[1])] != 0)):
        cur_pixel += dir_vec
    return cur_pixel

img_path = "test_files\\red_square.png"
img = cv2.imread(img_path)
h,w, channels = img.shape

# OPTIONS FOR FLOOD FILL
ignored_cells = [[255,255,255]]
min_area = 25
seed = (50,50)
floodflags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
flood_value = 255

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

            #Picks Random Starting Pixel
            possible_starts = np.where(roi != 0)
            rng = np.random.randint(0,len(possible_starts[0]))
            origin = np.array([possible_starts[0][rng],possible_starts[1][rng]])

            #ALGORTHM
            points = []
            dir_vec = np.array([1.0,0.0])

            new_point = CreateNewPoint(origin.astype(float), dir_vec, mask)
            print("ORIGIN", origin)
            print("DIR", dir_vec)
            print("NEW POINT", new_point)
            
            cv2.imshow("test", roi)
            cv2.waitKey(0)





    

