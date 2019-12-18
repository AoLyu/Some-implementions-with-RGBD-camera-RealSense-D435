import numpy as np 
import cv2 

img = cv2.imread('template.png')
img1 = cv2.GaussianBlur(img, (3, 3), 0)
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('Color', cv2.WINDOW_NORMAL)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) 

contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
xl = []
yl = []
rl = []
# print(len(contours))
for i,contour in enumerate(contours):
    if i==0:
        continue
    (x, y), radius = cv2.minEnclosingCircle(contours[i])
    print(radius)
    center = (int(x), int(y))
    radius = int(radius)
    # if 1 < radius and radius<6 :
    if 1:
        xl.append(int(x))
        yl.append(int(y))
        rl.append(radius)
# print(len(xl))

for i,x in enumerate(xl):
    cv2.circle(img, (xl[i],yl[i]), rl[i], (0, 255, 0), -1)

# yi = np.array(yl)*(-1)
# inds = np.argsort(yi)
# x_new = []
# y_new = []
# r_new = []
# for ind in inds:
#     x_new.append(xl[ind])
#     y_new.append(yl[ind])
#     r_new.append(rl[ind])

# for i,x in enumerate(x_new):
    # cv2.circle(img, (x_new[i],y_new[i]), r_new[i], (0, 0, 255), 3)

cv2.imshow("Color", img)
cv2.waitKey()
cv2.destroyAllWindows()