import cv2
import numpy as np
import time
import os

def merge(img_dir, h, l):
    img_name_book = sorted(os.listdir(img_dir))
    print (img_name_book)
    img_book = [cv2.resize(cv2.imread('{}/{}'.format(img_dir, img_name)), (256,256)) for img_name in img_name_book]
    for i in range(h):
        tmp = img_book[i*l]
        for j in range(1,l):
            tmp = np.hstack((tmp, img_book[i*l+j]))
        if i==0:
            res = tmp
        else:
            res = np.vstack((res, tmp))
    return res


if __name__ == '__main__':
    img_dir = './enhance'
    img = merge(img_dir, 3, 4)
    cv2.imwrite('merge_pintu.jpg', img)








