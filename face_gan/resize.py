import cv2

img = cv2.imread('./raw_images/smstyle1.jpeg')
img = cv2.resize(img,(1024,1024))
cv2.imwrite('./raw_images/smstyle1.jpeg', img)
