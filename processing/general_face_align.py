import cv2
import face_alignment
import numpy as np
import os
import math

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')

# https://github.com/davisking/dlib/blob/883101477d2485ae6e0e8499ec0eefb8382fcb5a/dlib/image_transforms/interpolation.h
# https://medium.com/@olga_kravchenko/generalized-procrustes-analysis-with-python-numpy-c571e8e8a421
# Procrustes Analysis


def get_position(desired_size, padding=0.25):
    # Average positions of face points 17-67  (reference shape)
    x = [0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
         0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
         0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
         0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
         0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
         0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
         0.553364, 0.490127, 0.42689]  # mean_face_shape_x

    y = [0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
         0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
         0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
         0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
         0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
         0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
         0.784792, 0.824182, 0.831803, 0.824182]  # mean_face_shape_y

    x, y = np.array(x), np.array(y)
    x = (x + padding) / (2 * padding + 1) * desired_size
    y = (y + padding) / (2 * padding + 1) * desired_size
    return np.array(list(zip(x, y)))


# def transformation_from_points(points1, points2):
#     points1 = points1.astype(np.float64)
#     points2 = points2.astype(np.float64)
#     c1 = np.mean(points1, axis=0)
#     c2 = np.mean(points2, axis=0)
#     points1 -= c1
#     points2 -= c2
#     s1 = np.std(points1)
#     s2 = np.std(points2)
#     points1 /= s1
#     points2 /= s2
#     # 注：points1和points2是np.matrix类型，*就相当于矩阵乘，即np.array的dot()和@
#     U, S, Vt = np.linalg.svd(points1.T * points2)
#     R = (U * Vt).T
#     return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),
#                       np.matrix([0., 0., 1.])])     # 3x3


def transformation_from_points(points1, points2):
    # points1：需要对齐的人脸关键点
    # points2：对齐的模板人脸(平均脸关键点)
    '''0 - 先确定是float数据类型 '''
    points1 = np.copy(points1).astype(np.float64)
    points2 = np.copy(points2).astype(np.float64)
    '''1 - 消除平移的影响 '''
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    '''2 - 消除缩放的影响 '''
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    '''3 - 计算矩阵M=BA^T；对矩阵M进行SVD分解；计算得到R '''
    # ||RA-B||; M=BA^T
    A = points1.T  # 2xN
    B = points2.T  # 2xN
    M = np.dot(B, A.T)
    U, S, Vt = np.linalg.svd(M)
    R = np.dot(U, Vt)
    '''4 - 构建仿射变换矩阵 '''
    s = s2 / s1
    sR = s * R
    c1 = c1.reshape(2, 1)
    c2 = c2.reshape(2, 1)
    T = c2 - np.dot(sR, c1)   # 模板人脸的中心位置减去需要对齐的中心位置（经过旋转和缩放之后）
    trans_mat = np.hstack([sR, T])  # 2x3
    return trans_mat


def get_landmarks(img):
    preds = fa.get_landmarks_from_image(img)  # list
    if preds is None:
        return None
    lms = preds[0].tolist()  # 68
    return np.array(lms)


def align_img(img_dir, save_dir, desired_size=256):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    files = list(os.listdir(img_dir))
    files = [file for file in files if (file.find('.jpg') != -1)]
    shapes = []
    imgs = []
    for file in files:
        I = cv2.imread(os.path.join(img_dir, file))
        shape = get_landmarks(I)
        if shape is None:
            print(file)
            continue
        imgs.append(I)
        shapes.append(shape[17:])

    mean_shape = get_position(desired_size)  # 模板脸(平均脸)

    for i, shape in enumerate(shapes):
        M = transformation_from_points(np.matrix(shape), np.matrix(mean_shape))
        img = cv2.warpAffine(imgs[i],
                             M[:2],   # [R|t]  2 x 3 仿射变换矩阵
                             dsize=(desired_size, desired_size),  # (cols, rows)
                             borderMode=cv2.BORDER_TRANSPARENT)
        # cv2.imwrite(os.path.join(save_dir, files[i]), img)
        # cx, cy = mean_shape[-20:].mean(0).astype(np.int32)  # 用标准脸的中心作为旋转中心
        shape = np.dot(shape, M[:, :2].T) + M[:, -1]  # Nx2
        cx, cy = np.mean(shape[-20:], axis=0).astype(np.int32)
        w = 160 // 2
        mouth = img[cy - w // 2: cy + w // 2, cx - w: cx + w, ...].copy()
        cv2.imwrite(os.path.join(save_dir, files[i]), mouth)

    print('Done!!')


def procrustes_analysis(shape, ref_shape):
    def get_rotation_matrix(theta):
        return np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])

    temp_sh = np.copy(shape)
    temp_ref = np.copy(ref_shape)
    temp_sh -= np.mean(temp_sh, axis=0)
    temp_ref -= np.mean(temp_ref, axis=0)

    a = np.sum((temp_sh * temp_ref).sum(1)) / np.linalg.norm(temp_ref) ** 2
    b = np.sum(temp_sh[:, 0] * temp_ref[:, 1] - temp_ref[:, 0] * temp_sh[:, 1]) / np.linalg.norm(temp_ref) ** 2
    theta = math.atan(b / max(a, 1e-9))
    scale = np.sqrt(a ** 2 + b ** 2)

    temp_sh /= round(scale, 1)
    rot_mat = get_rotation_matrix(round(theta, 2))
    aligned_shape = np.dot(rot_mat, temp_sh.T).T
    return aligned_shape


# def procrustes_analysis(shape, ref_shape):
#     def get_rotation_matrix(theta):
#         return np.array([[math.cos(theta), -math.sin(theta)],
#                          [math.sin(theta), math.cos(theta)]])
#
#     temp_sh = np.copy(shape)
#     temp_ref = np.copy(ref_shape)
#     temp_sh -= np.mean(temp_sh, axis=0)
#     temp_ref -= np.mean(temp_ref, axis=0)
#     temp_sh /= np.std(temp_sh)
#     temp_ref /= np.std(temp_ref)
#
#     a = np.sum((temp_sh * temp_ref).sum(1))
#     b = np.sum(temp_sh[:, 0] * temp_ref[:, 1] - temp_ref[:, 0] * temp_sh[:, 1])
#     theta = math.atan(b / max(a, 1e-9))
#
#     rot_mat = get_rotation_matrix(round(theta, 2))
#     aligned_shape = np.dot(rot_mat, temp_sh.T).T
#     return aligned_shape


# 普氏距离：对应点之间的欧氏距离之和 (形状差异的统计量度)
def procrustes_distance(shape, ref_shape):
    dist = np.linalg.norm(shape - ref_shape)
    # dist = np.sum(np.sqrt((shape[:, 0] - ref_shape[:, 0])**2 + (shape[:, 1] - ref_shape[:, 1])**2))
    return dist


def general_align_img(img_dir, save_dir, desired_size=256):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 获取人脸shape (面部关键点)
    files = list(os.listdir(img_dir))
    files = [file for file in files if (file.find('.jpg') != -1)]
    shapes = []
    imgs = []
    for file in files:
        I = cv2.imread(os.path.join(img_dir, file))
        shape = get_landmarks(I)
        if shape is None:
            continue
        imgs.append(I)
        shapes.append(shape[17:])

    # 2. 确定标准shape
    cur_dist = 0
    mean_shape = shapes[0]
    new_shapes = np.zeros(np.asarray(shapes).shape)
    while True:
        new_shapes[0] = mean_shape
        for i in range(1, len(shapes)):
            # M = transformation_from_points(shapes[i], mean_shape)  # 2 x 3
            # new_sh = (M @ np.c_[shapes[i], np.ones(len(shapes[i]))].T).T  # np.c_[] 增加一列
            # new_sh = (M[:2, :2] @ shapes[i].T).T + M[:2, -1].reshape(-1)
            new_sh = procrustes_analysis(shapes[i], mean_shape)
            new_shapes[i] = new_sh

        new_mean = np.mean(new_shapes, axis=0)
        new_dist = procrustes_distance(new_mean, mean_shape)
        if new_dist == cur_dist:   # 距离没有变化则退出迭代
            break

        # MM = transformation_from_points(new_mean, mean_shape)
        # new_mean = (MM @ np.c_[new_mean, np.ones(len(new_mean))].T).T  # np.c_[] 增加一列
        # new_mean = (MM[:2, :2] @ new_mean.T).T + MM[:2, -1].reshape(-1)
        new_mean = procrustes_analysis(new_mean, mean_shape)
        mean_shape = new_mean
        cur_dist = new_dist
    print(mean_shape)

    # 3. 实施人脸对齐
    for i, img in enumerate(imgs):
        M = transformation_from_points(shapes[i], mean_shape)
        new_img = cv2.warpAffine(img,
                                 M[:2],   # [R|t]  2 x 3 仿射变换矩阵
                                 (desired_size, desired_size),  # (cols, rows)
                                 borderMode=cv2.BORDER_TRANSPARENT)
        cv2.imwrite(os.path.join(save_dir, files[i]), new_img)
    print('Done!')



align_img('../src_faces/wh', 'tgt_faces/dyn')
# general_align_img('src_faces', 'tgt_faces/dyn')
