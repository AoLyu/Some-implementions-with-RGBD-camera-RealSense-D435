import numpy as np
import cv2 
import open3d as o3d

class Template():
    def __init__(self,img = None):
        # if img.all == None:
        #     raise Exception('template=Template(image)')
        self.img = img
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.cx = img.shape[1] / 2
        self.cy = img.shape[0] / 2
        self.realH = 210   #mm
        self.realW = 210   #mm
        self.s = (self.realH/self.height + self.realW/self.width)/2 /1000   # m/pix
        self.grayimg = None
        self.pointcloud1 = o3d.geometry.PointCloud()
        self.pointcloud2 = o3d.geometry.PointCloud()
        self.circleList = []
        if len(self.img.shape) == 3:
            self.grayimg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        elif len(self.img.shape) == 4:
            self.grayimg = cv2.cvtColor(self.img, cv2.COLOR_BGRA2GRAY)
        # self.grayimg = cv2.medianBlur(self.grayimg, 3)

    def getPt(self):
        circles = cv2.HoughCircles(self.grayimg, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=30, minRadius=0, maxRadius=22)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]: 
            self.circleList.append([i[0],i[1],i[2]])
        Pt1 = []
        Pt2 = []
        for j in range(len(self.circleList)):
            x = (self.circleList[j][0]-self.cx) *  self.s               # axis v   
            y = 0                          
            z = (self.circleList[j][1]-self.cy) *  self.s               # axis u
            Pt1.append([x,y,z])
        self.pointcloud1.points = o3d.utility.Vector3dVector(np.array(Pt1))
        self.pointcloud1.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        for j in range(len(self.circleList)):
            for ix in range(-1,2):
                for iz in range(-1,2):
                    x = (self.circleList[j][0]+ix-self.cx) *  self.s               # axis v   
                    y = 0                          
                    z = (self.circleList[j][1]+iz-self.cy) *  self.s               # axis u
                    Pt2.append([x,y,z])
        self.pointcloud2.points = o3d.utility.Vector3dVector(np.array(Pt2))
        self.pointcloud2.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        return self.circleList,self.pointcloud1,self.pointcloud2

    def getFeature(self):
        if len(self.circleList) ==0:
            self.getPt()
        # if n < 3:
        #     raise Exception('number must be bigger than 2!')
        n = 3  # k_nearpoint
        feature_list = []
    
        # id_list = []
        new_xyr = []
        kd_tree = o3d.geometry.KDTreeFlann(self.pointcloud1)
        [_, idx1, _] = kd_tree.search_knn_vector_3d(self.pointcloud1.points[0], len(self.pointcloud1.points))    #sorting the point cloud

        for point_i in idx1:
            array = np.array([])
            [_, idx, _] = kd_tree.search_knn_vector_3d(self.pointcloud1.points[point_i], n)
            for i in range(n):
                for j in range(n-1):
                    if j==i:
                        continue
                    for k in range(j+1,n):
                        if k == i:
                            continue
                        a = self.pointcloud1.points[idx[j]] - self.pointcloud1.points[idx[i]]
                        b = self.pointcloud1.points[idx[k]] - self.pointcloud1.points[idx[i]]
                        d1 = np.linalg.norm(a)
                        d2 = np.linalg.norm(b)
                        d3 = np.linalg.norm(b-a)
                        array = np.append(array,[d1,d2,d3])
            feature_list.append([self.pointcloud1.points[point_i],array])
            new_xyr.append(self.circleList[point_i])
        return feature_list ,new_xyr

if __name__=='__main__':
    img = cv2.imread('../template/template.png')
    template = Template(img)
    _,pointcloud1 ,pointcloud= template.getPt()
    o3d.visualization.draw_geometries([pointcloud])
    o3d.io.write_point_cloud('template.pcd',pointcloud)
    kd_tree = o3d.geometry.KDTreeFlann(pointcloud)
    [_, idx1, _] = kd_tree.search_knn_vector_3d(pointcloud.points[0], len(pointcloud.points))
    # print(idx1)

    
