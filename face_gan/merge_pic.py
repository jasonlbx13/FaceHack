import cv2
import numpy as np


k = 0
cache_h = []
for i in range(5):
    img_cache = cv2.imread("./results/{}.png".format(k))
    img_cache = cv2.resize(img_cache, None, fx=0.25, fy=0.25)
    for j in range(9):
        k += 1
        img = cv2.imread("./results/{}.png".format(k))
        img = cv2.resize(img, None, fx=0.25, fy=0.25)
        img_cache = np.hstack((img_cache, img))
    cache_h.append(img_cache)

img = cache_h[0]
for i in range(1,5):
    img = np.vstack((img, cache_h[i]))


cv2.imwrite('merge_img.jpg', img)
cv2.imshow("merged_img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
