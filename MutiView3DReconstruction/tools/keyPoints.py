import _init_paths
import numpy as np 
import cv2 
from scipy import optimize
from tools import rgbdTools

def getCircles(img):
    img1 = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    xl = []
    yl = []
    rl = []
    for i,contour in enumerate(contours):
        (x, y), radius = cv2.minEnclosingCircle(contours[i])
        center = (int(x), int(y))
        radius = int(radius)
        if 1 < radius and radius < 6 :
            xl.append(int(x))
            yl.append(int(y))
            rl.append(radius)
    yi = np.array(yl)
    inds = np.argsort(-yi)
    x_new = []
    y_new = []
    r_new = []
    for ind in inds:
        x_new.append(xl[ind])
        y_new.append(yl[ind])
        r_new.append(rl[ind])

    return x_new,y_new,r_new

def calculatePlane(cam,depth,xl,yl,rl):
    point_num = 0
    Points = []
    for i in range(len(xl)):
        if depth[yl[i],xl[i]] != 0:
            x,y,z = rgbdTools.getPosition(cam,depth,yl[i],xl[i])
            point = [x,y,z]
            Points.append(point)
            point_num +=1
        if point_num == 3:
            break
    Points = np.array(Points)
    a = (Points[1,1] - Points[0,1])*(Points[2,2] - Points[0,2]) - (Points[1,2] - Points[0,2])*(Points[2,1] - Points[0,1])
    b = (Points[1,2] - Points[0,2])*(Points[2,0] - Points[0,0]) - (Points[1,0] - Points[0,0])*(Points[2,2] - Points[0,2])
    c = (Points[1,0] - Points[0,0])*(Points[2,1] - Points[0,1]) - (Points[1,1] - Points[0,1])*(Points[2,0] - Points[0,0])
    d = 0 - (a * Points[0,0] + b*Points[0,1] + c*Points[0,2])

    xlist = []
    ylist = []
    zlist = []

    for num,x in enumerate(xl):
        if num > 5:
            # if 0 :
            break
        else:
            pix_l = pointInRadius(xl[num],yl[num],rl[num])
            for pix in pix_l:
                m1,n1 = pix
                if depth[m1,n1]!= 0:
                    x,y,z = rgbdTools.getPosition(cam,depth,m1,n1)
                    xlist.append(x)
                    ylist.append(y)
                    zlist.append(z)
    xarray = np.array(xlist)
    yarray = np.array(ylist)
    zarray = np.array(zlist)  
    
    r = optimize.leastsq(res,[a,b,c,d],args=(xarray,yarray,zarray))
    a,b,c,d = r[0]
    return a,b,c,d
  

def pointInRadius(x,y,r):
    pl = []
    for m in range(y-r,y+r+1):
        for n in range(x-r,x+r+1):
            if ((m-y)**2 + (n-x)**2)**0.5 <= r:
                pl.append((m,n))
    return pl

def res(p,xarray,yarray,zarray):
    a,b,c,d = p
    return abs(a*xarray+b*yarray+c*zarray+d)/(a**2+b**2+c**2)**0.5

class Plane:
    def __init__(self,a=0.1,b=0.1,c=0.1,d=0.1):
        self.a = a
        self.b = b
        self.c = c 
        self.d = d 
    def getParam(self,a,b,c,d):
        self.a = a
        self.b = b
        self.c = c 
        self.d = d 