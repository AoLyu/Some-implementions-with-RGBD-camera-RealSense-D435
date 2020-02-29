import cv2 as cv
import numpy as np 

planets = cv.imread('label.png')
cv.namedWindow('Color', cv.WINDOW_NORMAL)
gay_img = cv.cvtColor(planets, cv.COLOR_BGRA2GRAY)
# img = cv.medianBlur(gay_img, 7)  # 进行中值模糊，去噪点
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 50, param1=100, param2=30, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))
print(len(circles[0,:]))

for i in circles[0, :]:  # 遍历矩阵每一行的数据
    # cv.circle(planets, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # cv.circle(planets, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv.circle(planets, (i[0], i[1]), i[2], (0, 0, 255), -1)

cv.imshow("Color", planets)
cv.waitKey(0)
cv.destroyAllWindows()