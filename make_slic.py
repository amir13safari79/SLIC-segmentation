# import libraries and functions
import numpy as np
import cv2
from skimage import morphology
from find_gradient import find_gradient
from find_distributed_centroids import find_distributed_centroids

# find slic segmented img with desired k segment and alpha
def make_slic(img, k, alpha):
    # Resize the image to spend less time
    scale_percent = 0.6
    w = int(img.shape[1] * scale_percent)
    h = int(img.shape[0] * scale_percent)
    img = cv2.resize(img, (w, h))

    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find gradient of gray_img:
    gradient_img = find_gradient(gray_img)

    # start slic algorithm:
    # first using find_distributed_centroids finds centroid
    (s, centroid) = find_distributed_centroids(img, gradient_img, k)

    count = 0
    while (1):
        # produce a numpy array with shape = img that keep minimum cost for each img pixel
        # that assigned to its cluster center.
        # initially with inf value
        # and also produce label_img that keep label for each pixel initial with nan
        dist_img = np.full((img.shape[0], img.shape[1]), np.inf)
        label_img = np.zeros((img.shape[0], img.shape[1]))

        for i in range(centroid.shape[0]):
            ############# find feature space with (l,a,b,x,y) for each pixel #############
            x_c = centroid[i, 0]
            y_c = centroid[i, 1]

            x_min_range = max(0, x_c - int(s))
            x_max_range = min(img.shape[0], x_c + int(s))
            y_min_range = max(0, y_c - int(s))
            y_max_range = min(img.shape[1], y_c + int(s))

            lab_window = lab_img[x_min_range:x_max_range, y_min_range:y_max_range]
            x_range = np.arange(x_min_range, x_max_range)
            y_range = np.arange(y_min_range, y_max_range)
            x_window = np.transpose([x_range] * lab_window.shape[1])
            y_window = np.tile(y_range, (lab_window.shape[0], 1))

            # combine lab_window, x_window and y_window:
            feature_window = np.zeros((lab_window.shape[0], lab_window.shape[1], 5))
            feature_window[:, :, 0:3] = lab_window
            feature_window[:, :, 3] = x_window
            feature_window[:, :, 4] = y_window

            # convert 3d array feature_window to 2d feature_matrix:
            l_vector = feature_window[:, :, 0].flatten()
            a_vector = feature_window[:, :, 1].flatten()
            b_vector = feature_window[:, :, 2].flatten()
            x_vector = feature_window[:, :, 3].flatten()
            y_vector = feature_window[:, :, 4].flatten()

            feature_matrix = np.zeros((l_vector.shape[0], 5))
            feature_matrix[:, 0] = l_vector  # first column of feature_matrix is l_vector
            feature_matrix[:, 1] = a_vector
            feature_matrix[:, 2] = b_vector
            feature_matrix[:, 3] = x_vector
            feature_matrix[:, 4] = y_vector  # last column of feature_matrix is y_vector

            ############# find distance for each feature vector from i'th centroid #############
            centroid_vector = np.array([lab_img[x_c, y_c, 0],
                                        lab_img[x_c, y_c, 1],
                                        lab_img[x_c, y_c, 2],
                                        x_c,
                                        y_c])

            d_lab_matrix = np.sum((feature_matrix[:, 0:3] - centroid_vector[0:3]) ** 2, axis=1)
            d_xy_matrix = np.sum((feature_matrix[:, 3:5] - centroid_vector[3:5]) ** 2, axis=1)
            d_matrix = d_lab_matrix + alpha * d_xy_matrix

            ############# convert d_matrix to d_window for comparing with dist_img and assign label  #############
            dist_window = dist_img[x_min_range:x_max_range, y_min_range:y_max_range].copy()
            label_window = label_img[x_min_range:x_max_range, y_min_range:y_max_range].copy()
            d_window = d_matrix.reshape(x_range.shape[0], y_range.shape[0])

            label_window[dist_window > d_window] = i
            dist_window[dist_window > d_window] = d_window[dist_window > d_window]

            # update dist_img and label_img:
            dist_img[x_min_range:x_max_range, y_min_range:y_max_range] = dist_window.copy()
            label_img[x_min_range:x_max_range, y_min_range:y_max_range] = label_window.copy()

        ############# find new centroids  #############
        new_centroid = []
        for i in range(centroid.shape[0]):
            new_centroid_inds = np.where(label_img == i)
            new_x_c = int(np.mean(new_centroid_inds[0]))
            new_y_c = int(np.mean(new_centroid_inds[1]))
            new_centroid.append([new_x_c, new_y_c])

        new_centroid = np.array(new_centroid)

        ############# compute difference between new_centroid and centroid(previous centers) #############
        centroid_diff_tmp = centroid - new_centroid
        centroid_diff = np.hypot(centroid_diff_tmp[:, 0], centroid_diff_tmp[:, 1])

        if (np.amax(centroid_diff) < 30 or count > 10):
            break
        else:
            count += 1
            print(f'differ = {np.amax(centroid_diff)}')
            print(f'count = {count}')
            centroid = new_centroid

    ################# after finding label_img(label for each cluster) we have: ###############
    # using closing from morphology to Enforce connectivity:
    label_closing = morphology.closing(label_img, np.ones((30, 30), np.uint8))

    # from label_closing and using lapalcian find boundaries and apply to image
    label_boundaries = np.uint8(cv2.Laplacian(label_closing, -1, ksize=3)) > 1
    img_boundaries = cv2.merge([1 - label_boundaries, 1 - label_boundaries, 1 - label_boundaries])
    slic_img = img * img_boundaries

    return slic_img