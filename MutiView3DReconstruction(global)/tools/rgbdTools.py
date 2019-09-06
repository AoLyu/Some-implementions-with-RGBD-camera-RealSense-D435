import cv2
import numpy as np 
from scipy import optimize
import open3d as o3d 

class Camera:
    def __init__(self,fx,fy,cx,cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

def getPosition(cam,depth,m,n):                # input camera class ,rgb,depth array row , col,  return (x,y,z)
    z = depth[m,n]/1000
    x = ( n - cam.cx ) * z / cam.fx
    y = ( m - cam.cy ) * z / cam.fy
    return (x,y,z)

def getColor(rgb,m,n):                        # input camera class ,rgb,depth array row , col,  return (x,y,z)
    r = rgb[m,n,2]/255
    g = rgb[m,n,1]/255
    b = rgb[m,n,0]/255
    return [r,g,b]

