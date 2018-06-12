import Image
from timeit import default_timer as timer
import numpy as np
import cv2
import random

image1 = Image.Image("DSC08551.png.haraff.sift")
image2 = Image.Image("DSC08555.png.haraff.sift")
image1.read_data_from_file()
image2.read_data_from_file()

start_time = timer()
image1.distance_fast_all(image2)
end_time = timer()
result_time = end_time - start_time
print(result_time)
image1.find_pairs_quick_k(image2, 1)
image1.cohesion_pairs(450, 0.5)
image1.ransac("affine", 100, 10, 300, 100)

def draw_me_like_one_of_your_french_girls(image1, image2, image):
    iimagea = cv2.imread(image1)
    iimageb = cv2.imread(image2)

    new_image = np.concatenate((iimagea, iimageb), axis=0)

    for pair_calc in image.pairs:
        red = random.randrange(0, 255)
        blue = random.randrange(0, 255)
        green = random.randrange(0, 255)
        cv2.line(new_image, (int(pair_calc.point1.x), int(pair_calc.point1.y)),
                 (int(pair_calc.point2.x), int(iimagea.shape[0] + pair_calc.point1.y)), (red, green, blue), 1)

    cv2.imshow('image', new_image)
    cv2.imwrite('generated.png', new_image)
    cv2.waitKey(0)

draw_me_like_one_of_your_french_girls("DSC08551.png", "DSC08555.png", image1)
