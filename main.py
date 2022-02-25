# import libraries and functions:
import cv2
from make_slic import make_slic

# main body :
img = cv2.imread('org_img.jpg')
slic_64 = make_slic(img, 64, 0.3)
slic_256 = make_slic(img, 256, 0.3)
slic_1024 = make_slic(img, 1024, 0.3)
slic_2048 = make_slic(img, 2048, 0.3)

cv2.imwrite('result_imgs/slic_64_segment.jpg', slic_64)
cv2.imwrite('result_imgs/slic_256_segment.jpg', slic_256)
cv2.imwrite('result_imgs/slic_1024_segment.jpg', slic_1024)
cv2.imwrite('result_imgs/slic_2048_segment.jpg', slic_2048)