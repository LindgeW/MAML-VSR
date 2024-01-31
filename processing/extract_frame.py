import os
from torch.utils.data import DataLoader, Dataset
import time
import glob
import subprocess


class MyDataset(Dataset):
    def __init__(self):
        self.IN = 'video-high/'
        self.OUT = 'faces-small/'
        self.files = glob.glob(os.path.join(self.IN, 's*', '*.mpg'))  # mpg(mpeg)视频格式
        self.wav = 'audio'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]  # video-high/s1/bbcnzze.mpg
        _, ext = os.path.splitext(file)
        # 保存目录：faces-small/s1/bbcnzze
        dst = file.replace(self.IN, self.OUT).replace(ext, '')
        if not os.path.exists(dst):
            os.makedirs(dst)
            # cmd = 'ffmpeg -i {} -qscale:v 2 -r 25 {}/%d.jpg'.format(file, dst)   # 720x576
            # cmd = 'ffmpeg -i {} -qscale:v 2 -r 25 -s 360x288 {}/%d.jpg'.format(file, dst)
            cmd = 'ffmpeg -i {} -qscale:v 2 -r 25 -s 360x288 {}/%03d.jpg'.format(file, dst)  # 001.jpg 002.jpg ...
            os.system(cmd)
            # subprocess.run(cmd, shell=True)

        # wav = file.replace(self.IN, self.wav).replace(ext, '.wav')  # audio/s1/bbcnzze.wav
        # if not os.path.exists(wav):
        #     # 将输入的音频文件转换为单声道的、采样率为16kHz的无损PCM编码格式
        #     cmd = 'ffmpeg -y -i \'{}\' -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 \'{}\' '.format(file, wav)
        #     os.system(cmd)
            # subprocess.run(cmd, shell=True)
        return dst


if __name__ == '__main__':
    dataset = MyDataset()
    loader = DataLoader(dataset, num_workers=32, batch_size=128, shuffle=False, drop_last=False)
    tic = time.time()
    for i, _ in enumerate(loader):
        eta = (1.0 * time.time() - tic) / (i + 1) * (len(loader) - i)
        print('eta:{}'.format(eta / 3600.0))
