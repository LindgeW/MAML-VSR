import cv2
import numpy as np
import face_alignment
import sys

# http://matthewearl.github.io/2015/07/28/switching-eds-with-python/
# https://github.com/matthewearl/faceswap/blob/master/faceswap.py

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6


def get_landmarks(im):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')
    preds = fa.get_landmarks_from_image(im)  # list
    points_list = preds[0].tolist()  # 68
    return np.matrix(points_list)   # 注：不是np.array


# def annotate_landmarks(im, landmarks):
#     im = im.copy()
#     for idx, point in enumerate(landmarks):
#         pos = (point[0, 0], point[0, 1])
#         cv2.putText(im, str(idx), pos,
#                     fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
#                     fontScale=0.4,
#                     color=(0, 0, 255))
#         cv2.circle(im, pos, 3, color=(0, 255, 255))
#     return im


def get_face_mask(im, landmarks):
    def draw_convex_hull(im, points, color):
        points = cv2.convexHull(points.astype(np.int32))
        cv2.fillConvexPoly(im, points, color=color)

    im = np.zeros(im.shape[:2], dtype=np.float64)
    for group in OVERLAY_POINTS:
        draw_convex_hull(im, landmarks[group], color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    return im


def transformation_from_points(pt1, pt2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    # 注：pt1和pt2均为np.mat类型
    pt1 = pt1.astype(np.float64)
    pt2 = pt2.astype(np.float64)
    c1 = np.mean(pt1, axis=0)
    c2 = np.mean(pt2, axis=0)
    pt1 -= c1
    pt2 -= c2
    s1 = np.std(pt1)
    s2 = np.std(pt2)
    pt1 /= s1
    pt2 /= s2
    U, S, Vt = np.linalg.svd(pt1.T * pt2)   # (2, N) x (N, 2) -> (2, 2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T   # (1, 2)

    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])   # (3, 3)


# def transformation_from_points(pt1, pt2):
#     # 注：pt1和pt2均为np.array类型
#     pt1 = pt1.astype(np.float64)
#     pt2 = pt2.astype(np.float64)
#     c1 = np.mean(pt1, axis=0)
#     c2 = np.mean(pt2, axis=0)
#     pt1 -= c1
#     pt2 -= c2
#     s1 = np.std(pt1)
#     s2 = np.std(pt2)
#     pt1 /= s1
#     pt2 /= s2
#
#     '''计算矩阵M=BA^T；对矩阵M进行SVD分解；计算得到R '''
#     # ||RA-B||; M=BA^T
#     A = pt1.T  # 2xN
#     B = pt2.T  # 2xN
#     M = np.dot(B, A.T)
#     U, S, Vt = np.linalg.svd(M)
#     R = np.dot(U, Vt)
#
#     '''构建仿射变换矩阵 '''
#     s = s2 / s1
#     sR = s * R
#     c1 = c1.reshape(2, 1)
#     c2 = c2.reshape(2, 1)
#     T = c2 - np.dot(sR, c1)  # 模板人脸的中心位置减去需要对齐的中心位置（经过旋转和缩放之后）
#
#     trans_mat = np.hstack([sR, T])  # 2x3
#     return trans_mat


def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR, im.shape[0] * SCALE_FACTOR))
    lm = get_landmarks(im)
    print(im.shape)
    return im, lm


def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    # 利用变换矩阵M对图像进行旋转、平移、仿射等变换
    cv2.warpAffine(im,    # 输入图像
                   M[:2],   # 变换矩阵(旋转或平移) 2x3
                   (dshape[1], dshape[0]),  # 输出图像大小 (cols, rows)即(宽, 高)
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))


def run():
    im1, landmarks1 = read_im_and_landmarks(sys.argv[1])
    im2, landmarks2 = read_im_and_landmarks(sys.argv[2])

    M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])
    mask = get_face_mask(im2, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask], axis=0)

    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    cv2.imwrite('output.jpg', output_im)
    print('Done!!')


run()
