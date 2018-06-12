import Point as p
import Pair
import numpy as np
import scipy as sp
import scipy.spatial
import copy
import random


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

    # calculate distance between two points on the same image
    def calculate_physical_image_distance(self, point1, point2):
        p1 = np.array([point1.x, point1.y])
        p2 = np.array([point2.x, point2.y])
        a = np.array(p1)
        b = np.array(p2)
        a_1 = np.asmatrix(a)
        b_1 = np.asmatrix(b)
        return sp.spatial.distance.cdist(a_1, b_1, 'euclidean')

    # find pairs within given cohesion factor
    def cohesion_pairs(self, k, factor):
        consistent_pairs = []

        # matrix for both images with distances
        # between their own point from pairs
        image_1_matrix, image_2_matrix = self.pairs_distance_matrix()

        for i in range(0, len(self.pairs)):
            # list with k nearest pairs for both images
            # indexes are kept and are the same as self.pairs
            neighbourhood_image_1 = self.find_pair_point_neighbourhood(k, image_1_matrix[:, i])
            neighbourhood_image_2 = self.find_pair_point_neighbourhood(k, image_2_matrix[:, i])

            # common part of both nearest lists for images
            pairs_random_in_range = list(set(neighbourhood_image_1).intersection(neighbourhood_image_2))

            # factor of cohesion is counted from div between
            # common nearest pairs and all k pairs checked
            consistent_factor = len(pairs_random_in_range) / k

            if consistent_factor > factor:
                consistent_pairs.append(self.pairs[i])

        self.pairs = consistent_pairs
        print("After cohesion: ")
        print(len(self.pairs))

    # find indexes of k smallest values
    def find_pair_point_neighbourhood(self, k, values):
        return list(np.argpartition(values, k)[:k])

    # calculate distance between points on the same image
    # points are given from pairs
    def pairs_distance_matrix(self):
        image_1_crd = []
        image_2_crd = []

        for p in self.pairs:
            image_1_crd.append([p.point1.x, p.point1.y])
            image_2_crd.append([p.point2.x, p.point2.y])

        image_1_crd_array = np.array(image_1_crd)
        image_1_crd_matrix = np.asmatrix(image_1_crd_array)

        image_2_crd_array = np.array(image_2_crd)
        image_2_crd_matrix = np.asmatrix(image_2_crd_array)

        return sp.spatial.distance.cdist(image_1_crd_matrix, image_1_crd_matrix, 'euclidean'), \
               sp.spatial.distance.cdist(image_2_crd_matrix, image_2_crd_matrix, 'euclidean')

    # count distance between all features vectors
    # for both images
    def distance_fast_all(self, image):
        self.distances_to_another_image = sp.spatial.distance.cdist(self.feature_matrix, image.feature_matrix,
                                                                    'euclidean')

    # quick find pairs between points
    # according to feature matirix and given k
    # k = 1 will give nearest neighbour
    def find_pairs_quick_k(self, image, k):
        x = copy.deepcopy(self.distances_to_another_image)
        # cut features matrix to get k minimum values
        # for image_1 from rows
        min_dist_image_1 = np.argpartition(x, k, axis=0)
        min_dist_image_1 = min_dist_image_1[:k]
        # cut features matrix to get k minimum values
        # for image_2 from columns
        min_dist_image_2 = np.argpartition(x, k, axis=1)
        min_dist_image_2 = min_dist_image_2[:, :k]

        for i in range(0, self.points_size):
            img1 = min_dist_image_1[:, i]

            for b in range(0, k):
                img2 = min_dist_image_2[img1[b]]
                if i in img2[:]:
                    # add the pair to the list
                    # if nearest is symmetrical relation
                    self.pairs.append(Pair.Pair(self.points[i],
                                                image.points[img1[b]],
                                                self.distances_to_another_image[img1[b]][i]))
        print(self.pairs)
        print(len(self.pairs))

    # affine transformation
    def affine_transformation(self, chosen_pairs):
        X = np.array([[chosen_pairs[0].point1.x, chosen_pairs[0].point1.y, 1, 0, 0, 0],
                      [chosen_pairs[1].point1.x, chosen_pairs[1].point1.y, 1, 0, 0, 0],
                      [chosen_pairs[2].point1.x, chosen_pairs[2].point1.y, 1, 0, 0, 0],
                      [0, 0, 0, chosen_pairs[0].point1.x, chosen_pairs[0].point1.y, 1],
                      [0, 0, 0, chosen_pairs[1].point1.x, chosen_pairs[1].point1.y, 1],
                      [0, 0, 0, chosen_pairs[2].point1.x, chosen_pairs[2].point1.y, 1]])

        Y = np.array([[chosen_pairs[0].point2.x],
                      [chosen_pairs[1].point2.x],
                      [chosen_pairs[2].point2.x],
                      [chosen_pairs[0].point2.y],
                      [chosen_pairs[1].point2.y],
                      [chosen_pairs[2].point2.y]])

        X = np.asmatrix(X)
        Y = np.asmatrix(Y)
        det_X = np.linalg.det(X)

        if det_X != 0.0:
            X = np.linalg.inv(X)

            Z = X.dot(Y)
            Z = np.reshape(Z, (-1, 3))

            Z = np.append(Z, [[0, 0, 1]], axis=0)

            return Z

    # perspective transformation
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

        Y = np.array([[chosen_pairs[0].point2.x],
                      [chosen_pairs[1].point2.x],
                      [chosen_pairs[2].point2.x],
                      [chosen_pairs[3].point2.x],
                      [chosen_pairs[0].point2.y],
                      [chosen_pairs[1].point2.y],
                      [chosen_pairs[2].point2.y],
                      [chosen_pairs[3].point2.y]])

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

    # choose transformation type
    def transformation(self, method, chosen_pairs):
        if method == "perspective":
            return self.perspective_transformation(chosen_pairs)
        else:
            return self.affine_transformation(chosen_pairs)

    def ransac(self, method, error, min_val, max_val, iteration_value):

        # choose method
        chosen_pairs_number = 3
        if method == "perspective":
            chosen_pairs_number = 4

        # get coordinates for all points in pairs
        # image.point1 belongs to image1
        # image.point2 belongs to image2
        points_coordinates_image_1 = []
        points_coordinates_image_2 = []

        for p in self.pairs:
            image_1 = np.array([p.point1.x, p.point1.y, 1])
            image_1_matrix = np.asmatrix(image_1)
            image_1_matrix_transposed = np.transpose(image_1_matrix)
            points_coordinates_image_1.append(image_1_matrix_transposed)

            image_2 = np.array([p.point2.x, p.point2.y, 1])
            image_2_matrix = np.asmatrix(image_2)
            points_coordinates_image_2.append(image_2_matrix)

        # choose initial random ransac sample
        if len(self.pairs) >= chosen_pairs_number:
            chosen_pairs = random.sample(self.pairs, chosen_pairs_number)

            best_model = np.matrix
            best_model_score = 0
            final_pairs = []

            # iteration within given range
            for i in range(0, iteration_value):
                # calculate current model
                model = self.transformation(method, chosen_pairs)

                model_coordinates_for_best_model = []
                score = 0
                pairs_from_model = []

                # count all coordinates according to current model
                for point in points_coordinates_image_1:
                    model_coordinates_for_best_model.append(np.transpose(np.dot(model, point)))

                # count distance between point on image 2
                # and point calculated from model
                for p in range(0, len(self.pairs)):
                    c_1 = model_coordinates_for_best_model[p]
                    c_2 = points_coordinates_image_2[p]
                    dist = scipy.spatial.distance.cdist(c_1, c_2, 'euclidean')
                    if dist.item(0) < error:
                        score += 1
                        pairs_from_model.append(self.pairs[p])

                # if model is better, replace data
                if score > best_model_score:
                    best_model = model
                    best_model_score = score
                    final_pairs = pairs_from_model

                # find new set of pairs
                chosen_pairs = self.chose_new_set(min_val, max_val, chosen_pairs)

            print(best_model_score)
            print(best_model)

            # change pairs to final post-ransac set
            self.pairs = final_pairs
        else:
            print("Wrong pairs set!")

    def chose_new_set(self, min_val, max_val, previously_chosen):
        new_set = []

        # find replacement for all previously counted pairs
        for p in previously_chosen:
            unique_added = True
            while unique_added:
                # get random pair
                pair = random.choice(self.pairs)

                # check if is in given distance between the old one
                if min_val < self.calculate_physical_image_distance(p.point1, pair.point1) < max_val \
                        and min_val < self.calculate_physical_image_distance(p.point2, pair.point2) < max_val:
                    new_set.append(pair)

                # chceck if you did not append the same pair twice
                list_size_prev = len(new_set)
                new_set = list(set(new_set))
                list_size_after = len(new_set)
                unique_added = list_size_prev == list_size_after

        return list(new_set)
