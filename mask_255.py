import cv2
import os

path = '/home/slz/PycharmProjects/2D_covid/val/mask1/'
mask_ = os.listdir(path)
for name in mask_:
    mask = cv2.imread(path + name, 0)
    cv2.imwrite(path + name, mask * 255)
