import numpy as np
import matplotlib.pylab as plt
import scipy.spatial as spatial
import triangle as tr
import cv2

#Parameters
fileName = "test_files/colors.png"

#Image Processing
img = cv2.imread(fileName)
unique_color_data = np.unique(img.reshape((1, -1, 3))[0,:],axis=0, return_counts=True)
blacklist = np.array([[255,255,255], [0,0,0], [255,0,0]]) #NOTE, cv2 has pixels in bgr order

for color in unique_color_data[0]:
    #Skips Colors On Blacklist
    if(np.any(np.all(blacklist == color, axis=-1))):
        print(color)
        continue

    coords = np.where(np.all(img == color, axis=-1))
    
    for i in range(len(coords[0])):
        img[coords[0][i],coords[1][i]] = [0,0,0]

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