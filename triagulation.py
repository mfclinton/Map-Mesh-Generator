import numpy as np
import matplotlib.pylab as plt
import scipy.spatial as spatial
import cv2

# Preparing File
fileName = "red_circle.png"
img = cv2.imread(fileName)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
points = np.column_stack(np.where(gray == 0))
points = np.insert(points, 2, 0, axis=1)

# Select Points
numPoints = 20
chosenPoints = []

for i in range(numPoints):
    points = points[points[:,2].argsort()]
    bestPoint = points[-1]
    for p in points:
        p[-1] += spatial.distance.euclidean(bestPoint, p)
    bestPoint[-1] = -9999 #temp solution, should remove


    chosenPoints.append([bestPoint[0], bestPoint[1]])

chosenPoints = np.array(chosenPoints)
print(chosenPoints)

# print(np.insert(points, 2, 0, axis=1))
# randomPoints = points[np.random.choice(points.shape[0], 100, replace = False)]
# print(randomPoints)

# plot triangle
tri = spatial.Delaunay(chosenPoints)
plt.triplot(chosenPoints[:,0], chosenPoints[:,1], tri.simplices)
# plt.plot(points[:,0], points[:,1], 'o') # plot points
plt.show()










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