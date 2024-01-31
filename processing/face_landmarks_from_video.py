import os
import cv2
import numpy as np
import glob
import face_alignment
from skimage import io
import json
import torch
import time

# 批量处理视频帧的面部关键点


def get_imgs_from_video(video, ext='jpg', RGB=False):
    frames = []
    if os.path.isdir(video):
        frames = sorted(glob.glob(os.path.join(video, '*.{}'.format(ext))),
                        key=lambda x: int(x.split(os.path.sep)[-1].split('.')[0]))
        frames = [cv2.imread(f) for f in frames]
    else:
        cap = cv2.VideoCapture(video)
        # cap.set(cv2.CAP_PROP_FPS, 25)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度  640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度  480
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率 30fps
        frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_num / fps   # sec
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # 视频的编码  22
        # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
        print(width, height, fps, frame_num, duration, fourcc)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

    frames = np.array(frames)
    if RGB:
        return frames[..., ::-1]
    else:
        return frames


def get_landmarks(path, save_dir='landmarks'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # if torch.cuda.is_available():
    #     device = 'cuda'
    # else:
    #     device = 'cpu'

    frames = get_imgs_from_video(path)
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='sfd')
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='blazeface')
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu')
    for i, frame in enumerate(frames):
        preds = fa.get_landmarks_from_image(frame)   # list
        points_list = preds[0].tolist()   # 68
        assert len(points_list) == 68
        for j, one_point in enumerate(points_list):
            x, y, _ = one_point
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # cv2.putText(frame, str(j+1), (int(x)-5, int(y)+5), fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=0.3, color=(0, 0, 0), thickness=1)
        lm_img = os.path.join(save_dir, f'{path.split(".")[0]}_lm{i + 1}.jpg')
        cv2.imwrite(lm_img, frame)

        lm_pts = os.path.join(save_dir, f'{path.split(".")[0]}_lm{i+1}' + '.txt')
        np.savetxt(lm_pts, points_list)
    print('Done')


def get_batch_landmarks(path):
    frames = get_imgs_from_video(path)
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='sfd')
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='blazeface')
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device="cuda")
    bs = 10
    t1 = time.time()
    for b in [0, 1, 2, 3, 4, 5, 6, 7]:
        batch = np.stack(frames[bs*b: bs*(b+1)])   # (B, H, W, C)
        batch = batch.transpose(0, 3, 1, 2)   # (B, C, H, W)
        print('starting ...', len(batch))
        preds = fa.get_landmarks_from_batch(torch.tensor(batch))
        print('ending ...')
        for i, pred in enumerate(preds):
            if len(pred) > 0:
                np.savetxt(f'{path.split(".")[0]}_lm{i+1}.txt', pred)
    t2 = time.time()
    print(f'time cost: {t2 - t1} secs.')

# get_landmarks('grid_test.mpg')
get_batch_landmarks('bbizzn.mpg')