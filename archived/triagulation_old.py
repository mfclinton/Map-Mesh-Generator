import numpy as np
import matplotlib.pylab as plt
import scipy.spatial as spatial
import triangle as tr
import cv2

import time
import threading
from concurrent.futures import ThreadPoolExecutor

#Helper Functions

#Calculates the accuracy of the PSLG
def CalculateAccuracy(img, coords, points, line_segments):
    # print(points)
    # print(line_segments)
    score = 0
    for c in coords:
        intersections = 0
        for l in line_segments:
            p1 = points[l[0]]
            p2 = points[l[1]]

            # quick terminate
            if(not (p1[0] <= c[0] <= p2[0] or p2[0] <= c[0] <= p1[0])):
                continue

            #divide by zero error
            if(p1[0] - p2[0] == 0):
                if(p1[1] <= c[1] <= p2[1] or p2[1] <= c[1] <= p1[1]):
                    intersections += 1
                continue

            x_intercept = (c[0] - p1[0])*(p1[1] - p2[1])/(p1[0] - p2[0]) + p1[1]

            #intersects the line
            if(c[1] <= x_intercept):
                if(x_intercept != p1[1] and x_intercept != p2[1]):
                    intersections += 1
                #special corner interection case
                elif((x_intercept == p1[1] and p1[0] > p2[0]) or (x_intercept == p2[1] and p2[0] > p1[0])):
                    intersections += 1

        if(intersections % 2 != 1):
            continue

        # img[c[0],c[1]] = [0,0,0]
        score += 1

    return score

def ThreadedCalculateAccuracy(img, coords, points, line_segments, numThreads):
    results = []
    with ThreadPoolExecutor(max_workers=numThreads) as executor:
        chunks = np.array_split(coords,numThreads)
        results = executor.map(CalculateAccuracy,[img]*numThreads,chunks,[points]*numThreads, [line_segments]*numThreads)
    
    score = 0
    for value in results:
        score += value
    return score

def GetIntersections():
    print("")

#Parameters
fileName = "test_files/colors.png"

#Image Processing
img = cv2.imread(fileName) #(row, column) order
unique_color_data = np.unique(img.reshape((1, -1, 3))[0,:],axis=0, return_counts=True)
blacklist = np.array([[255,255,255], [0,0,0]]) #NOTE, cv2 has pixels in bgr order

#testing purposes
startTime = time.time()

for color in unique_color_data[0]:
    #Skips Colors On Blacklist
    if(np.any(np.all(blacklist == color, axis=-1))):
        continue

    coords = np.where(np.all(img == color, axis=-1))
    coords = np.array((coords[0],coords[1])).T

    area = len(coords)

    #building triangle
    points = []
    line_segments = [(0,1),(0,2),(1,2)]

    #rng trial
    # rng = np.random.randint(0,area,size=3)
    # for i in range(len(rng)):
    #     points.append((coords[rng[i]][0],coords[rng[i]][1]))

    #rng iterative brute force
    rng = np.random.randint(0,area,size=3)
    for i in range(len(rng)):
        points.append([coords[rng[i]][0],coords[rng[i]][1]])

    iterations = 3
    for i in range(iterations):
        iterationTime = time.time()
        rng = np.random.randint(0,3)
        point = points[rng]
        # score = CalculateAccuracy(img, coords, points, line_segments)
        score = ThreadedCalculateAccuracy(img, coords, points, line_segments, 20)
        points.pop(rng)
        for c in coords:
            if((c == points[0]).all() or (c == points[1]).all()):
                continue
            points.append(c)
            # temp_score = CalculateAccuracy(img, coords, points, line_segments)
            temp_score = ThreadedCalculateAccuracy(img, coords, points, line_segments, 10)
            if(temp_score > score):
                point = c
                score = temp_score
            #   
            points.pop()
        points.append(point)
        # print(points)
        print(score/area)
        print("iteration_time: %f"%(time.time() - iterationTime))
        print("total_time: %f"%(time.time() - startTime))

    #brute force trial *never finishes*
    # points = []
    # score = 0
    # for c1 in coords:
    #     for c2 in coords:
    #         if((c2 == c1).all()):
    #             continue
    #         for c3 in coords:
    #             if((c3 == c1).all() or (c3 == c2).all()):
    #                 continue
    #             temp_points = [c1,c2,c3]
    #             temp_score = CalculateAccuracy(img, coords, temp_points, line_segments)
    #             if(temp_score > score):
    #                 score = temp_score
    #                 points = temp_points

    # calculate scores
    # score = CalculateAccuracy(img, coords, points, line_segments)
    # score = ThreadedCalculateAccuracy(img, coords, points, line_segments, 16)
    print("%s : %f" % (color, score/area))

    for p in points:
        img[p[0],p[1]] = [255,255,255]

#random-no-threading: .05
print("time: %f"%(time.time() - startTime))


cv2.imshow("image", img)
cv2.waitKey(0)














#--------------------------------

# print(tr.get_data('face'))

# # Preparing File
# fileName = "test_files/europe.png"
# img = cv2.imread(fileName)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# points = np.column_stack(np.where(gray == 0))
# dog = points
# points = np.insert(points, 2, 0, axis=1)

# # Select Points
# numPoints = 200
# chosenPoints = []

# for i in range(numPoints):
#     points = points[points[:,2].argsort()]
#     bestPoint = points[-1]
#     for p in points:
#         dist = spatial.distance.euclidean(bestPoint, p)
#         p[-1] += dist
#     bestPoint[-1] = -9999 #temp solution, should remove


#     chosenPoints.append([bestPoint[0], bestPoint[1]])

# chosenPoints = np.array(chosenPoints)

# # print(np.insert(points, 2, 0, axis=1))
# # randomPoints = points[np.random.choice(points.shape[0], 100, replace = False)]
# # print(randomPoints)

# # plot triangle
# A = dict(vertices = chosenPoints)
# B = tr.triangulate(A)
# tr.compare(plt, A, B)
# # plt.triplot(chosenPoints[:,0], chosenPoints[:,1], tri.simplices)
# # plt.plot(points[:,0], points[:,1], 'o') # plot points
# plt.show()










# OLD CODE

# print(randomPoints)
# points = np.concatenate(points, np.zeros(len(points)))
# print(points)

# borders = cv2.inRange(img, black_value, black_value)
# borders = cv2.bitwise_not(borders) # makes the borders black
# print(borders)
# cv2.imshow("Borders", borders)
# cv2.waitKey(0)








# points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# img = cv2.imread("red_square.png")
# height, width, channels = img.shape
# numPoints = 80
# size = 99
# points = np.random.rand(numPoints,2) * size

# filterArray = []
# for p in points:
#     # print(img[int(p[0]),int(p[1])])
#     if(img[int(p[0]),int(p[1])][0] != 255):
#         filterArray.append(p)
#         print(p)

# tri = spatial.Delaunay(filterArray)

# plt.triplot(points[:,0], points[:,1], tri.simplices)
# plt.plot(points[:,0], points[:,1], 'o')
# plt.show()