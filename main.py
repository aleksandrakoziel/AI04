import Image
from timeit import default_timer as timer
import numpy as np

image1 = Image.Image("DSC08551.png.haraff.sift")
image2 = Image.Image("DSC08555.png.haraff.sift")
image1.read_data_from_file()
image2.read_data_from_file()

#pairs = image1.knn(image1.points[1413], image2, 5)
#
# for p in pairs:
#     print(p)
#
# pairs[0].visualize_pair("DSC08551.png", "DSC08555.png")
#
# image1.count_distances(image2)
# print(image1.distances_to_another_image)

# image1.find_all_pairs(image2)
# k_pairs = image1.find_k_nearest_pairs(image1.pairs, image1.pairs[1], 5, 100)
# print(k_pairs)
# print(" ")
# random_pairs = image1.find_random_in_range(image1.pairs, image1.pairs[1],
start_time = timer()
image1.distance_fast_all(image2)
end_time = timer()
result_time = end_time - start_time
print(result_time)
image1.find_pairs_quick_k(image2, 1)
print(len(image1.pairs))
# image1.pairs[107].visualize_pair("DSC08551.png", "DSC08555.png")
# transform = image1.perspective_transformation(image1.pairs[:4])
# print(transform)
image1.draw_all_pairs("DSC08551.png", "DSC08555.png")