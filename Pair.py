import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch


class Pair:
    def __init__(self, point1, point2, distance):
        self.point1 = point1
        self.point2 = point2
        self.distance = distance

    def __str__(self):
        return "First " + str(self.point1) \
               + "\nSecond " + str(self.point2) \
               + "\nDistance = " + str(self.distance)

    def visualize_pair(self, image1, image2):
        img1 = mpimg.imread(image1)
        img2 = mpimg.imread(image2)

        x1 = self.point1.x
        y1 = self.point1.y

        x2 = self.point2.x
        y2 = self.point2.y


        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img1)
        ax1.plot(x1, y1, 'r*')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(img2)
        ax2.plot(x2, y2, 'r*')

        con = ConnectionPatch(xyA=(x2, y2), xyB=(x1, y1), coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(con)
        fig.savefig('result.png')



    def find_k_pair_neighbours(self):
        pass