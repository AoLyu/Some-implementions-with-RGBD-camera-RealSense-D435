import numpy as np 
import open3d as o3d 
import math 

def extractFeatures(pointCloud,o_xyr,n = 3):
    if n < 3:
        print('number must be bigger than 2!')
        n = 3
    feature_list = []
    
    # id_list = []
    new_xyr = []
    kd_tree = o3d.geometry.KDTreeFlann(pointCloud)
    [_, idx1, _] = kd_tree.search_knn_vector_3d(pointCloud.points[0], len(pointCloud.points))
    # for idz in idx1:
    #     # id_list.append(idz)
    #     new_xyr.append(xyr[idz])
    for point_i in idx1:
        array = np.array([])
        [_, idx, _] = kd_tree.search_knn_vector_3d(pointCloud.points[point_i], n)
        for i in range(n):
            for j in range(n-1):
                if j==i:
                    continue
                for k in range(j+1,n):
                    if k == i:
                        continue
                    a = pointCloud.points[idx[j]] - pointCloud.points[idx[i]]
                    b = pointCloud.points[idx[k]] - pointCloud.points[idx[i]]
                    d1 = np.linalg.norm(a)
                    d2 = np.linalg.norm(b)
                    d3 = np.linalg.norm(b-a)
                    array = np.append(array,[d1,d2,d3])
        feature_list.append([pointCloud.points[point_i],array])
        new_xyr.append(o_xyr[point_i])
    return feature_list ,new_xyr
   

def fastMatch(sFeatureList,tFeatureList):

    match_list = []
    ind_list = []  
    small_L =[]
    for ind1 in range(len(sFeatureList)):
        small_dis = 10
        sPoint = sFeatureList[ind1][0]
        tPt = None
        ind2 = None
        for indt,tPoint in enumerate(tFeatureList):
            current_dis = dis(sFeatureList[ind1][1],tPoint[1])
            if current_dis < small_dis:
                small_dis = current_dis
                tPt = tPoint[0]
                ind2 = indt
        # print(small_dis)
        small_L.append(small_dis)

        if small_dis < 0.012:
            match_list.append([sPoint,tPt])
            ind_list.append((ind1,ind2))
    # small_L.sort()
    # for lll in small_L:
    #     print(lll)
    array1 = np.array(match_list)[:,0]
    array2 = np.array(match_list)[:,1]

    Trans_init = rigid_transform_3D(array1, array2)
    return Trans_init,ind_list


def icp(source,target,trans_init):
    threshold = 0.001
    reg_p2p = o3d.registration.registration_icp(source, target, threshold, trans_init,
                                                o3d.registration.TransformationEstimationPointToPoint()) 
    return reg_p2p.transformation

def colored_icp(source , target , trans_init = np.identity(4) ,voxel_size = 0.005 ,iter_num =15):
    source_down = o3d.geometry.voxel_down_sample(source, voxel_size)
    target_down = o3d.geometry.voxel_down_sample(target, voxel_size)
    o3d.geometry.estimate_normals(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size, max_nn=30))
    o3d.geometry.estimate_normals(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size, max_nn=30))
    result_icp = o3d.registration.registration_colored_icp(
            source_down, target_down, voxel_size, trans_init,
            o3d.registration.ICPConvergenceCriteria(relative_fitness = 1e-6,
                                                    relative_rmse = 1e-6,
                                                    max_iteration = iter_num))
    return result_icp.transformation


def dis(arr1,arr2):
    if len(arr1)!=len(arr2):
        print('length of two array are not equal!')
    else:
        dis = np.linalg.norm(arr1 - arr2,ord=1)
        return dis 

def rigid_transform_3D(A, B):
    assert len(A) == len(B)
    N = A.shape[0]
    A = A.T
    B = B.T  
    mu_A = np.mean(A, axis=1).reshape(3,1)
    mu_B = np.mean(B, axis=1).reshape(3,1)
    a = A - mu_A
    b = B - mu_B
    H = (a).dot(b.T)  
    U, S, Vt = np.linalg.svd(H)
    mat_temp = U.dot(Vt)
    det = np.linalg.det(mat_temp)
    datM = np.array([[1,0,0],[0,1,0],[0,0,det]])
    R = (Vt.T).dot(datM).dot(U.T)
    t = mu_B- R.dot(mu_A)
    T  = np.zeros((4,4))
    T[:3,:3] = R
    for i in range(3):
        T[i,3] = t[i]
    T[3,3] = 1
    return T


