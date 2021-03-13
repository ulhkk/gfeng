# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
    if correlation:
        corr_data = np.corrcoef(data.T)
        ##print(corr_data)
        eigenvectors, eigenvalues, _ = np.linalg.svd(corr_data)        
        ##print("using correlation")
        
    else:
        cov_data = np.cov(data.T)
        eigenvectors, eigenvalues, _ = np.linalg.svd(cov_data)
        ##print("using cov")
        

    eigenvalues = np.sqrt(eigenvalues)
    ##print("eigenvalues", eigenvalues)
    ##print(eigenvectors)
    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main():
    # 指定点云路径
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    # 加载原始点云
    point_cloud_pynt = PyntCloud.from_file("/home/gfeng/gfeng_ws/point_cloud_processing/ch1_introduction/hw1/ply_data/airplane/test/1.ply")
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])
    print('The size of points is: ',points.shape)
    # 用PCA分析点云主方向
    w, v = PCA(points,False)
    point_cloud_vector = v[:, 0] #点云主方向对应的向量
    # print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # TODO: 此处只显示了点云，还没有显示PCA
    # draw line set 
    point_center = np.mean(points,axis = 0)
    points_ = [point_center,(point_center +v[:,0] * w[0] * 100 )  ,( point_center + v[:,1]  * w[1] * 100)]
    lines_ = [[0,1],[0,2]]
    colors_ = [[1,0,0],[0,1,0]]
    
    
    line_set = o3d.geometry.LineSet()
    line_set.lines = o3d.utility.Vector2iVector(lines_)
    line_set.colors = o3d.utility.Vector3dVector(colors_)
    line_set.points = o3d.utility.Vector3dVector(points_)
    #o3d.visualization.draw_geometries([point_cloud_o3d, line_set])
    
    #
    '''dim_reduction = v[:,:2]
    pcl_2d = np.dot(np.array(points), dim_reduction)
    plt.scatter(pcl_2d[:,0], pcl_2d[:,1])
    plt.show()'''


    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []
    # 作业2
    # 屏蔽开始
    for i in range(points.shape[0]):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i], 20)
        knn_points = np.asarray(point_cloud_o3d.points)[idx, :]
        _, v_knn_points = PCA(knn_points)
        normals.append(v_knn_points[:,-1])

    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数
    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()
