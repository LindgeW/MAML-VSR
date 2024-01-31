import os
import cv2
import numpy as np
import glob
import numba
import face_alignment
from threading import Thread
# from multiprocessing import Process
import time
import dlib


# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='sfd')
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='blazeface')
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda')   # 默认sfd


def run(files, fa):
    for img_name, save_name in files:
        if os.path.exists(save_name) and os.path.getsize(save_name) > 0:
            continue
        t1 = time.time()
        frame = cv2.imread(img_name)
        points_list = fa.get_landmarks_from_image(frame)  # list
        if points_list is None:   # 未检测到关键点
            print(f'Bad File: {img_name}')
        else:
            points = points_list[0].tolist()  # 68
            if len(points) == 68:
                np.savetxt(save_name, points, fmt='%d')
                t2 = time.time()
                print(f'time cost: {t2 - t1}s')


def get_landmarks(frame_root):
    assert os.path.exists(frame_root)
    # faces = glob.glob(os.path.join(frame_root, '*', '*.jpg'))
    faces = []
    for root, dirs, files in os.walk(frame_root):
        for f in files:
            if f.endswith('.jpg'):
                faces.append(os.path.join(root, f))

    data = [(name, name.replace('.jpg', '.xy')) for name in faces]  # 或者.txt
    print(len(data))

    records = []
    gids = [0, 1, 2]
    bs = len(data) // len(gids)   # 每个gpu分的数据量
    # 每个gpu分配一个face alignment
    fas = [face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:' + str(gpu_id)) for gpu_id in
           gids]  # 默认sfd
    for i, fa in enumerate(fas):  # 平分数据，分给每个gpu处理
        if i == len(gids) - 1:
            bs = len(data)
        th = Thread(target=run, args=(data[:bs], fa,))
        # th = Process(target=video_process, args=(vid_dir,))
        data = data[bs:]
        th.start()
        records.append(th)

    for th in records:
        th.join()

    print('Done!!!')


face_root = r'./faces-small'
get_landmarks(face_root)
