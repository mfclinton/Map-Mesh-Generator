import threading

class Worker(threading.Thread):
    def __init__(self, b_box, color):
        threading.Thread.__init__(self)
        self.b_box = b_box
        self.color = color
    
    def run(self):
        print("Starting " + str(self.color))
        ProcessBoundingBox(self.b_box, self.color)

def ProcessBoundingBox(b_box, color):
    print(color)
    pass