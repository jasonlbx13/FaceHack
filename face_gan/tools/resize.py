import cv2
import argparse

parser = argparse.ArgumentParser(description='resize pictures')
parser.add_argument('img_dir', help='image direction')
parser.add_argument('img_height', help='image height')
parser.add_argument('img_weight', help='image weight')
args = parser.parse_args()

img = cv2.imread(args.img_dir)
img = cv2.resize(img,(int(args.img_weight), int(args.img_height)))
cv2.imwrite(args.img_dir, img)
print ('done')
