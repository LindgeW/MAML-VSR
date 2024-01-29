import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from lipnet import LipNet
import cv2
import numpy as np
from jiwer import cer, wer


PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'
DEVICE = 'cuda:0'


class MetaLearner(nn.Module):
    def __init__(self, vocab_size, dropout=0.5):
        super(MetaLearner, self).__init__()
        self.model = LipNet(vocab_size, dropout)

    def forward(self, x, y, xlens, ylens):
        log_probs = self.model(x).log_softmax(dim=-1)  # (T, B, V)
        return F.ctc_loss(log_probs, y, xlens.reshape(-1), ylens.reshape(-1), zero_infinity=True)

    def decoding(self, x):
        with torch.no_grad():
            logit = self.model(x).transpose(0, 1)  # (B, T, v)
            return logit.data.cpu().argmax(dim=-1)


def HorizontalFlip(video_imgs, p=0.5):
    # (T, C, H, W)
    if np.random.random() < p:
        video_imgs = video_imgs[...,::-1].copy()
    return video_imgs


class GRIDDataset(Dataset):
    def __init__(self, data, phase='train'):
        if isinstance(data, str):
            self.dataset = self.get_data_file(data)
        else:
            self.dataset = data
        print(len(self.dataset))
        self.phase = phase
        self.char_dict = [PAD] + [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                  'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']    # 28 
        self.max_vid_len = 75
        self.max_txt_len = 50

    # 得到所有task文件目录的list (一个包含多个不同task的文件夹)
    def get_data_file(self, root_path):
        # GRID\LIP_160x80\lip\s1
        dataset = []
        unseen_spk = ['s1', 's2', 's20', 's21', 's22']
        for spk in os.listdir(root_path):  # 根目录下的speaker目录
            if spk in unseen_spk:
                continue
            data_path = os.path.join(root_path, spk)
            for fn in os.listdir(data_path):  # 1000
                path = os.path.join(data_path, fn)
                if len(os.listdir(path)) == 75:
                    dataset.append(path)
        return dataset

    def load_video(self, fn):
        files = os.listdir(fn)
        files = list(filter(lambda f: f.endswith('.jpg'), files))
        files = sorted(files, key=lambda f: int(os.path.splitext(f)[0]))
        array = [cv2.imread(os.path.join(fn, f), 0) for f in files]  # 单通道
        # array = list(filter(lambda im: im is not None, array))
        array = [cv2.resize(img, (128, 64)) for img in array]
        array = np.stack(array, axis=0)[:, None].astype(np.float32)  # TCHW  C=1
        return array / 255.

    def load_txt(self, fn):
        with open(fn, 'r', encoding='utf-8') as f:
            txt = [line.strip().split(' ')[2] for line in f]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
        raw_txt = ' '.join(txt).upper()
        return np.asarray([self.char_dict.index(c) for c in raw_txt])

    def padding(self, array, max_len):
        if len(array) >= max_len:
            return array[:max_len]
        return np.concatenate([array, np.zeros([max_len - len(array)] + list(array[0].shape), dtype=array[0].dtype)])

    def __getitem__(self, idx):
        item = self.dataset[idx]
        vid = self.load_video(item)
        if self.phase == 'train':
            vid = HorizontalFlip(vid, 0.5)
        txt_path = item.replace('lip', 'align_txt') + '.align'
        txt = self.load_txt(txt_path)
        vid_len = min(len(vid), self.max_vid_len)
        txt_len = min(len(txt), self.max_txt_len)
        vid = self.padding(vid, self.max_vid_len)
        txt = self.padding(txt, self.max_txt_len)
        return dict(vid=torch.FloatTensor(vid).transpose(0, 1),  # (C, T, H, W)
                    txt=torch.LongTensor(txt),
                    vid_lens=torch.tensor(vid_len),
                    txt_lens=torch.tensor(txt_len))

    def __len__(self):
        return len(self.dataset)


def train(dataset, lr=1e-4, epochs=100, batch_size=50, model_path=None):
    model = MetaLearner(28).to(DEVICE)
    print(sum(param.numel() for param in model.parameters()))

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu')
        states = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(states)
        print('loading weights ...')
    model.train()
    print(model)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_iters//10, num_training_steps=num_iters)
    for ep in range(epochs):
        for i, batch_data in enumerate(data_loader):
            inputs = batch_data['vid'].to(DEVICE)
            targets = batch_data['txt'].to(DEVICE)
            input_lens = batch_data['vid_lens'].to(DEVICE)
            target_lens = batch_data['txt_lens'].to(DEVICE)
            optimizer.zero_grad()
            loss = model(inputs, targets, input_lens, target_lens)
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            if (i + 1) % 10 == 0:
                print("Epoch {}, Iteration {}, loss: {:.4f}".format(ep + 1, i + 1, loss.data.item()), flush=True)
        savename = 'vanilla_iter_{}.pt'.format(ep + 1)
        savedir = os.path.join('checkpoints', 'grid3')
        if not os.path.exists(savedir): os.makedirs(savedir)
        torch.save({'model': model.state_dict()}, os.path.join(savedir, savename))
        print(f'Saved to {savename}.')


def adapt(model_path, data_path, lr=1e-4, epochs=100, batch_size=50):
    model = MetaLearner(28).to(DEVICE)
    print(sum(param.numel() for param in model.parameters())/1e6, 'M')
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu')
        states = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(states, strict=False)
        print('loading weights ...')
    #model.model.reset_params()
    model.train()
    print(model)
    spk_data = [os.path.join(data_path, fn) for fn in os.listdir(data_path)]
    adapt_data = spk_data[:500]  # half
    #dataset = GRIDDataset(adapt_data[:20])  # 1min
    #dataset = GRIDDataset(adapt_data[:60])  # 3min
    dataset = GRIDDataset(adapt_data[:100])  # 5min
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optimizer = optim.AdamW(model.model.adapter.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    #optimizer = optim.AdamW([*model.model.adanet.parameters(), model.model.sc], lr=lr, betas=(0.9, 0.98), eps=1e-9)
    #optimizer = optim.AdamW([*model.model.adanet.parameters(), *model.model.adanet2.parameters(), model.model.sc], lr=lr, betas=(0.9, 0.98), eps=1e-9)
    #optimizer = optim.AdamW([*model.model.fc.parameters(), *model.model.gru2.parameters()], lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_iters//10, num_training_steps=num_iters)
    for ep in range(epochs):
        for i, batch_data in enumerate(data_loader):
            inputs = batch_data['vid'].to(DEVICE)
            targets = batch_data['txt'].to(DEVICE)
            input_lens = batch_data['vid_lens'].to(DEVICE)
            target_lens = batch_data['txt_lens'].to(DEVICE)
            model.zero_grad()
            loss = model(inputs, targets, input_lens, target_lens)
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            if (i + 1) % 10 == 0:
                print("Epoch {}, Iteration {}, loss: {:.4f}".format(ep + 1, i + 1, loss.data.item()), flush=True)
        savename = 'vanilla_iter_{}.pt'.format(ep + 1)
        savedir = os.path.join('checkpoints', 'adapt_grid')
        if not os.path.exists(savedir): os.makedirs(savedir)
        torch.save({'model': model.state_dict()}, os.path.join(savedir, savename))
        print(f'Saved to {savename}.')


@torch.no_grad()
def evaluate(model_path, data_path, batch_size=50):
    model = MetaLearner(28).to(DEVICE)
    # checkpoint = torch.load(opt.load, map_location=lambda storage, loc: storage)
    checkpoint = torch.load(model_path, map_location='cpu')
    states = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(states)
    model.eval()
    print(model)
    dataset = GRIDDataset([os.path.join(data_path, fn) for fn in os.listdir(data_path)][500:], phase='test')
    print(len(dataset))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=5)
    preds = []
    refs = []
    for batch_data in data_loader:
        vid_inp = batch_data['vid'].to(DEVICE)
        tgt_txt = batch_data['txt'].to(DEVICE)
        output = model.decoding(vid_inp)
        pred = []
        gold = []
        for out, tgt in zip(output, tgt_txt):
            pred.append(''.join([dataset.char_dict[i] for i in torch.unique_consecutive(out).tolist() if i != 0]))
            gold.append(''.join([dataset.char_dict[i] for i in tgt.tolist() if i != 0]))
        #print(pred, gold)
        preds.extend(pred)
        refs.extend(gold)
    test_wer, test_cer = wer(refs, preds), cer(refs, preds)
    print('JIWER wer: {:.4f}, cer: {:.4f}'.format(test_wer, test_cer))


if __name__ == '__main__':
    #dataset = GRIDDataset(r'E:\GRID\LIP_160_80\lip')
    #train(dataset, lr=1e-4, epochs=50, batch_size=50)
    #train(dataset, lr=5e-5, epochs=50, batch_size=50, model_path='checkpoints/grid2/vanilla_iter_50.pt')

    #evaluate('checkpoints/grid3/vanilla_iter_32.pt', r'E:\GRID\LIP_160_80\lip\s1', batch_size=50)
    #evaluate('checkpoints/grid3/vanilla_iter_32.pt', r'E:\GRID\LIP_160_80\lip\s2', batch_size=50)
    #evaluate('checkpoints/grid3/vanilla_iter_32.pt', r'E:\GRID\LIP_160_80\lip\s20', batch_size=50)
    #evaluate('checkpoints/grid3/vanilla_iter_32.pt', r'E:\GRID\LIP_160_80\lip\s22', batch_size=50)


    adapt('checkpoints/grid3/vanilla_iter_32.pt', r'E:\GRID\LIP_160_80\lip\s22', lr=1e-4, epochs=50, batch_size=10)
    #evaluate('checkpoints/adapt_grid/vanilla_iter_50.pt', r'E:\GRID\LIP_160_80\lip\s1', batch_size=50)
    #evaluate('checkpoints/adapt_grid/vanilla_iter_50.pt', r'E:\GRID\LIP_160_80\lip\s2', batch_size=50)
    #evaluate('checkpoints/adapt_grid/vanilla_iter_50.pt', r'E:\GRID\LIP_160_80\lip\s20', batch_size=50)
    evaluate('checkpoints/adapt_grid/vanilla_iter_50.pt', r'E:\GRID\LIP_160_80\lip\s22', batch_size=50)

