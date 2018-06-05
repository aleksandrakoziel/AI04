import numpy as np

class Point:

    def __init__(self, x, y, a, b, c, features_vector):
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.c = c
        a = np.array(features_vector)
        a_1 = np.asmatrix(a)
        self.features_vector = a_1
        self.visited = False
        self.sum_features_vector = sum(features_vector)

    def __str__(self):
        return "Point x=" + str(self.x) \
               + " y=" + str(self.y) \
               + " a=" + str(self.a) \
               + " b=" + str(self.b) \
               + " c=" + str(self.c) \
               + " features vector=" + str(self.features_vector)
