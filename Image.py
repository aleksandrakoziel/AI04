import Point as p
import Pair
import numpy as np
import scipy as sp
import scipy.spatial
import copy
import random
from timeit import default_timer as timer


class Image:

    def __init__(self, file):
        self.points = np.array
        self.points_size = 0
        self.feature_size = 0
        self.feature_matrix = np.array
        self.file = file
        self.distances_to_another_image = np.matrix
        self.pairs = []

    def read_data_from_file(self):
        with open(self.file, mode='r', encoding='utf-8') as data:
            self.feature_size = int(data.readline())
            self.points_size = int(data.readline())

            self.load_data_to_point(data)

    def load_data_to_point(self, data):
        x = []
        y = []
        for i in range(0, self.points_size):
            data_line = data.readline()
            point_data = [float(x) for x in data_line.split(' ') if x != '']
            x.append(p.Point(point_data[0],
                             point_data[1],
                             point_data[2],
                             point_data[3],
                             point_data[4],
                             point_data[5:]))
            y.append(point_data[5:])
        a = np.array(x)
        self.points = a
        b = np.array(y)
        b_1 = np.asmatrix(b)
        self.feature_matrix = b_1

    def calculate_distance(self, point1, point2):
        p1 = np.array(point1.features_vector)
        p2 = np.array(point2.features_vector)
        a = np.array(p1)
        b = np.array(p2)
        a_1 = np.asmatrix(a)
        b_1 = np.asmatrix(b)
        distance = sp.spatial.distance.cdist(a_1, b_1, 'euclidean')
        return distance

    def knn(self, point, image, k):
        distances = []
        for i in range(0, image.points_size):
            distances.append([i, self.calculate_distance(point, image.points[i])])
        distances.sort(key=lambda tup: tup[1])
        neighbours = distances[:k]
        neighbours_as_pairs = []
        for n in neighbours:
            neighbours_as_pairs.append(Pair.Pair(point, image.points[n[0]], n[1]))
        return neighbours_as_pairs

    def nearest_neighbour(self, point):
        distances = []
        for i in range(0, self.points_size):
            distances.append([i, self.calculate_distance(point, self.points[i])])
        neighbour = min(distances)
        return neighbour

    def nearest_neighbour_pair(self, point, first_image_points, second_image_points):
        distances = []
        for i in range(0, len(second_image_points)):
            distances.append([i, self.calculate_distance(point, second_image_points[i])])
        distances.sort(key=lambda tup: tup[1])
        neighbour = second_image_points[distances[0][0]]
        neighbour_pair = self.nearest_neighbour(neighbour)
        if point is neighbour_pair:
            return Pair.Pair(point, neighbour, distances[0][1])
        else:
            return None

    def nearest_neighbour_pair_quick(self, image, k):
        pairs = []
        for i in range(0, len(self.points)):
            print("trying")
            column = self.distances_to_another_image[:, i]
            column = column.tolist()
            point_1_row = column.index(max(column))
            row = self.distances_to_another_image[point_1_row, :]
            row = row.tolist()
            point_2_column = row.index(max(row))
            if i is point_2_column:
                distance = self.distances_to_another_image[point_1_row][i]
                pairs.append(Pair.Pair(self.points[i], image.points[point_1_row], distance))
                print("pair added")

        return pairs

    def k_pairs_in_range(self, point, image, min, max, k):
        neighbours = []
        i = 0
        while i < k:
            neighbour = random.choice(image.points)
            distance = self.calculate_distance(point, neighbour)
            if min <= distance <= max:
                neighbours.append(Pair.Pair(point, neighbour, distance))
                i += 1
        return neighbours

    def find_k_nearest_pairs(self, pairs, pair, k, parameter):
        pairs_copy = copy.copy(pairs)
        neighbour1 = []
        neighbour2 = []
        pairs_copy.remove(pair)

        for p in pairs_copy:
            distance = self.calculate_distance(pair.point1, p.point1)
            neighbour1.append([distance, p])
            distance = self.calculate_distance(pair.point2, p.point2)
            neighbour2.append([distance, p])

        neighbour1.sort(key=lambda tup: tup[0])
        neighbour2.sort(key=lambda tup: tup[0])

        nearest_n2 = []
        a = 0
        while len(nearest_n2) < parameter:
            nearest_n2.append(neighbour2[a][1])
            a += 1

        i = 0
        j = 0
        result = []
        while i < k:
            if j < parameter:
                if neighbour1[j][1] is nearest_n2:
                    i += 1
                    result.append(neighbour1[j][i])
                j += 1
            else:
                break

        return result

    def find_random_in_range(self, min, max, k):
        pairs_copy = copy.copy(self.pairs)
        neighbour1 = []
        neighbour2 = []
        pair = random.choice(my_sequence)
        pairs_copy.remove(pair)

        for p in pairs_copy:
            distance = self.calculate_distance(pair.point1, p.point1)
            if min <= distance <= max:
                neighbour1.append(p)
            distance = self.calculate_distance(pair.point2, p.point2)
            if min <= distance <= max:
                neighbour2.append(p)

        # common part of neighbours for both images
        pairs_random_in_range = list(set(neighbour1) - (set(neighbour2) - set(neighbour1)))

        return pairs_random_in_range

    def real_distance_1(self, pair1, pair2):
        distance = pair1.point1.x*pair2.point1.x - pair1.point1.y*pair2.point1.y

    def distance_fast_all(self, image):
        self.distances_to_another_image = sp.spatial.distance.cdist(self.feature_matrix, image.feature_matrix,
                                                                    'euclidean')

    # def find_pairs_quick(self, image):
    #     x = copy.deepcopy(self.distances_to_another_image)
    #     min_dist_image_1 = x.min(axis=0)
    #     min_dist_image_2 = x.min(axis=1)
    #     min_dist_indexes_1 = x.argmin(0)
    #     min_dist_indexes_2 = x.argmin(1)
    #     print("zzzzzzzzzzzzzzzzzz 1")
    #     print(min_dist_image_1)
    #     print(min_dist_indexes_1)
    #     print("zzzzzzzzzzzzzzzzzzzzzz 2")
    #
    #     print(min_dist_image_2)
    #     print(min_dist_indexes_2)
    #
    #     for i in range(0, self.points_size):
    #         img1 = min_dist_indexes_1[i]
    #         img2 = min_dist_indexes_2[img1]
    #         dist_1 = min_dist_image_1[i]
    #         dist_2 = min_dist_image_1[img1]
    #
    #         if dist_1 == dist_2:
    #             print("pair")
    #             self.pairs.append(Pair.Pair(self.points[i], image.points[img1], min_dist_image_1[i]))
    #     print(self.pairs)

    def find_pairs_quick_k(self, image, k):
        x = copy.deepcopy(self.distances_to_another_image)
        min_dist_image_1 = np.argpartition(x, k, axis=0)
        min_dist_image_1 = min_dist_image_1[:k]
        min_dist_image_2 = np.argpartition(x, k, axis=1)
        min_dist_image_2 = min_dist_image_2[:, :k]

        for i in range(0, self.points_size):
            img1 = min_dist_image_1[:, i]

            print(img1)
            for b in range(0, k):
                img2 = min_dist_image_2[img1[b]]
                if i in img2[:]:
                    self.pairs.append(Pair.Pair(self.points[i],
                                                image.points[img1[b]],
                                                self.distances_to_another_image[img1[b]][i]))
        print(self.pairs)

    def draw_all_pairs(self, image1, image2):
        for p in self.pairs:
            p.visualize_pair(image1, image2)

    def affine_transformation(self, chosen_pairs):
        X = np.array([[chosen_pairs[0].point1.x, chosen_pairs[0].point1.y, 1, 0, 0, 0],
                      [chosen_pairs[1].point1.x, chosen_pairs[1].point1.y, 1, 0, 0, 0],
                      [chosen_pairs[2].point1.x, chosen_pairs[2].point1.y, 1, 0, 0, 0],
                      [0, 0, 0, chosen_pairs[0].point1.x, chosen_pairs[0].point1.y, 1],
                      [0, 0, 0, chosen_pairs[1].point1.x, chosen_pairs[1].point1.y, 1],
                      [0, 0, 0, chosen_pairs[2].point1.x, chosen_pairs[2].point1.y, 1]])

        Y = np.array([[chosen_pairs[0].point1.x],
                      [chosen_pairs[1].point1.x],
                      [chosen_pairs[2].point1.x],
                      [chosen_pairs[0].point1.y],
                      [chosen_pairs[1].point1.y],
                      [chosen_pairs[2].point1.y]])

        X = np.asmatrix(X)
        Y = np.asmatrix(Y)
        det_X = np.linalg.det(X)
        print(det_X)

        if det_X != 0.0:
            X = np.linalg.inv(X)

            Z = X.dot(Y)
            Z = np.reshape(Z, (-1, 3))

            Z = np.append(Z, [[0, 0, 1]], axis=0)

            return Z

    def perspective_transformation(self, chosen_pairs):
        X = np.array([[chosen_pairs[0].point1.x, chosen_pairs[0].point1.y, 1, 0, 0, 0,
                       -(chosen_pairs[0].point2.x * chosen_pairs[0].point1.x),
                       -(chosen_pairs[0].point2.x * chosen_pairs[0].point1.y)],
                      [chosen_pairs[1].point1.x, chosen_pairs[1].point1.y, 1, 0, 0, 0,
                       -(chosen_pairs[1].point2.x * chosen_pairs[1].point1.x),
                       -(chosen_pairs[1].point2.x * chosen_pairs[1].point1.y)],
                      [chosen_pairs[2].point1.x, chosen_pairs[2].point1.y, 1, 0, 0, 0,
                       -(chosen_pairs[2].point2.x * chosen_pairs[2].point1.x),
                       -(chosen_pairs[2].point2.x * chosen_pairs[2].point1.y)],
                      [chosen_pairs[3].point1.x, chosen_pairs[3].point1.y, 1, 0, 0, 0,
                       -(chosen_pairs[3].point2.x * chosen_pairs[3].point1.x),
                       -(chosen_pairs[3].point2.x * chosen_pairs[3].point1.y)],
                      [0, 0, 0, chosen_pairs[0].point1.x, chosen_pairs[0].point1.y, 1,
                       -(chosen_pairs[0].point2.y * chosen_pairs[0].point1.x),
                       -(chosen_pairs[0].point2.y * chosen_pairs[0].point1.y)],
                      [0, 0, 0, chosen_pairs[1].point1.x, chosen_pairs[1].point1.y, 1,
                       -(chosen_pairs[1].point2.y * chosen_pairs[1].point1.x),
                       -(chosen_pairs[1].point2.y * chosen_pairs[1].point1.y)],
                      [0, 0, 0, chosen_pairs[2].point1.x, chosen_pairs[2].point1.y, 1,
                       -(chosen_pairs[2].point2.y * chosen_pairs[2].point1.x),
                       -(chosen_pairs[2].point2.y * chosen_pairs[2].point1.y)],
                      [0, 0, 0, chosen_pairs[3].point1.x, chosen_pairs[3].point1.y, 1,
                       -(chosen_pairs[3].point2.y * chosen_pairs[3].point1.x),
                       -(chosen_pairs[3].point2.y * chosen_pairs[3].point1.y)]])

        Y = np.array([[chosen_pairs[0].point1.x],
                      [chosen_pairs[1].point1.x],
                      [chosen_pairs[2].point1.x],
                      [chosen_pairs[3].point1.x],
                      [chosen_pairs[0].point1.y],
                      [chosen_pairs[1].point1.y],
                      [chosen_pairs[2].point1.y],
                      [chosen_pairs[3].point1.y]])

        X = np.asmatrix(X)
        Y = np.asmatrix(Y)
        det_X = np.linalg.det(X)
        print(det_X)

        if det_X != 0.0:
            X = np.linalg.inv(X)

            Z = X.dot(Y)
            Z = np.insert(Z, 8, 1)
            Z = np.reshape(Z, (-1, 3))

            return Z

    def ransac(self, method):

        pass
