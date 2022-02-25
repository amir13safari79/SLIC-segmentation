# import libraries:
import numpy as np
import cv2


def find_distributed_centroids(img, gradient_img, k):
    img_area = img.shape[0] * img.shape[1]
    s = np.sqrt(img_area // k)  # s is difference between two cluster in initial state
    init_x_centroid = np.arange(s // 2, img.shape[0], s)
    init_x_centroid = np.uint32(init_x_centroid)

    init_y_centroid = np.arange(s // 2, img.shape[1], s)
    init_y_centroid = np.uint32(init_y_centroid)

    # join x_centroid and y_centroid:
    init_centroid = []
    for i in range(init_x_centroid.shape[0]):
        for j in range(init_y_centroid.shape[0]):
            init_centroid.append([init_x_centroid[i], init_y_centroid[j]])
    init_centroid = np.array(init_centroid)

    # move centroid to the points with lowest gradint with 5*5 window arount
    # each cluster center:
    centroid = []
    for i in range(init_centroid.shape[0]):
        init_x_c = init_centroid[i, 0]
        init_y_c = init_centroid[i, 1]
        window_gradient = gradient_img[init_x_c - 2:init_x_c + 3, init_y_c - 2:init_y_c + 3]
        min_inds = np.where(window_gradient == np.amin(window_gradient))
        rand_ind = np.random.randint(min_inds[0].shape)  # for find pixel with minimum gradient randomly

        # find x_c and y_c and append to centroid
        x_c = min_inds[0][rand_ind] + init_x_c - 2
        y_c = min_inds[1][rand_ind] + init_y_c - 2
        centroid.append([x_c[0], y_c[0]])

    centroid = np.array(centroid)

    return (s, centroid)