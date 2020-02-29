import numpy as np 
import cv2 

for i in range(6):
    img = cv2.imread('depth_{}.png'.format(i),-1)
    img = np.array(img)
    min_value = img.min()
    max_value = img.max()
    print(min_value,max_value)
    lenth = max_value - min_value
    img = img - min_value
    img = img / lenth * 255
    img = np.asanyarray(img, np.uint8)
    img = cv2.applyColorMap(img,cv2.COLORMAP_JET)
    cv2.imwrite('color_{}.png'.format(i),img)
