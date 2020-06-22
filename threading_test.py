import threading
from concurrent.futures import ThreadPoolExecutor
from numba import jit, cuda
import numpy as np

# @jit(nopython=True)
# def take(val):
#     for i in range(100000):
#         print("hello")
#     return 420

# results = []
# with ThreadPoolExecutor(max_workers=8) as executor:
#     results = executor.map(take, range(3))
#     print("I LOVE DOGS")

# print("EXITED LOL")
# for value in results:
#     print(value)

points = np.array([[0,1],[1,2],[3,3],[5,5]])

print(points[0:1 + 1])

for i in range(1+1, 3):
    print('lol')
# print(np.mean(points, axis=0))
# print(np.multiply(np.mean(points, axis=0)))

# print(np.multiply(points[:,0], points[:,1]))
# print(np.mean(np.multiply(points[:,0], points[:,1])))
# print(np.square(points[:,0]))
# print(np.mean(np.square(points[:,0])))