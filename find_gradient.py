# import libraries:
import numpy as np
import cv2


def find_gradient(gray_img):
    x_kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    y_kernel = x_kernel.T
    x_gradient = cv2.filter2D(gray_img, -1, x_kernel)
    y_gradient = cv2.filter2D(gray_img, -1, y_kernel)
    gradient_img = np.hypot(x_gradient, y_gradient)

    return gradient_img