import threading
from concurrent.futures import ThreadPoolExecutor
from numba import jit, cuda

@jit(nopython=True)
def take(val):
    for i in range(100000):
        print("hello")
    return 420

results = []
with ThreadPoolExecutor(max_workers=8) as executor:
    results = executor.map(take, range(3))
    print("I LOVE DOGS")

print("EXITED LOL")
for value in results:
    print(value)
