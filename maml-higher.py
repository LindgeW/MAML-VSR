# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import cv2
import numpy as np
from collections import defaultdict
from transformers import get_cosine_schedule_with_warmup
import higher
from seq2seq import Seq2Seq
from collections import namedtuple
from jiwer import cer, wer

PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'
DEVICE = 'cuda:0'
seed = 37


class MetaLearner(nn.Module):
    def __init__(self):
        super(MetaLearner, self).__init__()
        Args = namedtuple('Args', ['in_channel', 'hidden_dim', 'enc_layers', 'dec_layers', 'head_num', 'ffn_ratio', 'drop_attn', 'tgt_vocab_size', 'tgt_pad_idx'])
        opt = Args(1, 256, 3, 3, 8, 4, 0.1, 30, 0)
        self.model = Seq2Seq(opt)

    def forward(self, x, y, xlens, ylens):
        return self.model(x, y, xlens, ylens)

    def decoding(self, x, xlens, bos_id, eos_id, max_dec_len=80):
        with torch.no_grad():
            return self.model.beam_search_decoding(x, xlens, bos_id, eos_id, max_dec_len)


class MAMLDataset(Dataset):
    def __init__(self, data_path, batch_size):
        self.file_list = self.get_file_list(data_path)
        self.batch_size = batch_size  # 任务的数量，相当于每训练1个step就要训练完这么多个task

    # 得到所有task文件目录的list (一个包含多个不同task的文件夹)
    def get_file_list(self, data_path):
        # GRID\LIP_160x80\lip\s1
        unseen_spk = ['s1', 's2', 's20', 's21', 's22']
        return [os.path.join(data_path, fn) for fn in os.listdir(data_path) if fn not in unseen_spk]  # 根目录下的speaker目录

    # 返回一个task的数据，包括sup set和qry set
    def get_one_task_data(self, idx):  # one batch task data
        # GRID\LIP_160x80\lip\s1\bbaf4p
        # task_data_path = np.random.choice(self.file_list, size=1, replace=False)  # 不重复地采样
        task_data = []
        for fn in os.listdir(self.file_list[idx]):  # 1000
            path = os.path.join(self.file_list[idx], fn)
            if len(os.listdir(path)) == 75:
                task_data.append(path)
        task = Task(task_data)
        #sup_data = task.sample_data_batch(self.batch_size)
        #qry_data = task.sample_data_batch(self.batch_size)
        sampled_data = task.sample_data_batch(2*self.batch_size)
        #sup_data = {k: v[:self.batch_size] for k, v in sampled_data.items()}  
        #qry_data = {k: v[self.batch_size:] for k, v in sampled_data.items()}  
        sup_data = dict((k, v[:self.batch_size]) for k, v in sampled_data.items())
        qry_data = dict((k, v[self.batch_size:]) for k, v in sampled_data.items())
        return sup_data, qry_data

    def __getitem__(self, idx):
        return self.get_one_task_data(idx)

    def __len__(self):
        return len(self.file_list)


class Task(object):  # Speaker
    def __init__(self, data):
        # GRID\LIP_160x80\lip\s1\bbaf4p
        self.data = data
        '''
        self.data = []  # speaker目录下的视频目录
        for fn in os.listdir(dir_path):
            path = os.path.join(dir_path, fn.decode('utf-8'))
            if len(os.listdir(path)) > 0:
                self.data.append(path)
        '''
        # qry_size = int(len(self.data) * 0.2)
        # self.qry_set = self.data[:qry_size]
        # self.sup_set = self.data[qry_size:]
        self.char_dict = [PAD] + [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] + [EOS, BOS]  # 30
        self.max_vid_len = 75
        self.max_txt_len = 50

    def seq_data_batch(self, batch_size):
        for batch_paths in DataLoader(self.data, batch_size=batch_size, shuffle=False):
            vids = []
            txts = []
            vid_lens = []
            for path in batch_paths:
                vid = self.load_video(path)
                txt_path = path.replace('lip', 'align_txt') + '.align'
                txt = self.load_txt(txt_path)
                vid_lens.append(min(len(vid), self.max_vid_len))
                vids.append(self.padding(vid, self.max_vid_len))
                txts.append(self.padding(txt, self.max_txt_len))
            vids = np.stack(vids, axis=0)  # (B, T, C, H, W)
            txts = np.stack(txts, axis=0)
            yield dict(vid=torch.FloatTensor(vids),  # (B, T, C, H, w)
                       vid_lens=torch.tensor(vid_lens),  # (B, )
                       txt=torch.LongTensor(txts))  # (B, L)

    def sample_data_batch(self, batch_size):
        vids = []
        txts = []
        vid_lens = []
        txt_lens = []
        batch_paths = np.random.choice(self.data, size=batch_size, replace=False)  # 不重复地采样
        for path in batch_paths:
            vid = self.load_video(path)
            txt_path = path.replace('lip', 'align_txt') + '.align'
            txt = self.load_txt(txt_path)
            vid_lens.append(min(len(vid), self.max_vid_len))
            txt_lens.append(min(len(txt), self.max_txt_len))
            vids.append(self.padding(vid, self.max_vid_len))
            txts.append(self.padding(txt, self.max_txt_len))
        vids = np.stack(vids, axis=0)  # (B, T, C, H, W)
        txts = np.stack(txts, axis=0)
        return dict(vid=torch.FloatTensor(vids),  # (B, T, C, H, w)
                    txt=torch.LongTensor(txts),  # (B, L)
                    vid_lens=torch.tensor(vid_lens),  # (B, )
                    txt_lens=torch.tensor(txt_lens))  # (B, )

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
        return np.asarray([self.char_dict.index(BOS)] + [self.char_dict.index(c) for c in raw_txt] + [self.char_dict.index(EOS)])

    def padding(self, array, max_len):
        if len(array) >= max_len:
            return array[:max_len]
        return np.concatenate([array, np.zeros([max_len - len(array)] + list(array[0].shape), dtype=array[0].dtype)])


# meta training过程
def maml_train(dataset, inner_lr, meta_lr, num_iters=10000, meta_batch=5, model_path=None):
    meta_model = MetaLearner().to(DEVICE)
    if model_path is not None and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        states = checkpoint['model'] if 'model' in checkpoint else checkpoint
        meta_model.load_state_dict(states)
        print('loading model ...')
    meta_model.train()
    print(meta_model)
    task_loader = DataLoader(dataset, batch_size=meta_batch, shuffle=True, num_workers=meta_batch)
    meta_optimizer = optim.AdamW(meta_model.parameters(), lr=meta_lr, betas=(0.9, 0.98), eps=1e-9)
    # lr_scheduler = get_cosine_schedule_with_warmup(meta_optimizer, num_warmup_steps=num_iters//10, num_training_steps=num_iters)
    for i in range(num_iters):
        for sup_data, qry_data in task_loader:
            meta_losses = []
            meta_optimizer.zero_grad()
            inner_optimizer = optim.SGD([meta_model.model.encoder.code], lr=inner_lr)
            #inner_optimizer = optim.SGD(meta_model.model.parameters(), lr=inner_lr)
            for k in range(sup_data['vid'].shape[0]):
                with higher.innerloop_ctx(meta_model, inner_optimizer, copy_initial_weights=False) as (task_model, task_optimizer):
                    for _ in range(3):  # 内循环更新次数 (e.g., 1-5)
                        inputs = sup_data['vid'][k].to(DEVICE)
                        targets = sup_data['txt'][k].to(DEVICE)
                        input_lens = sup_data['vid_lens'][k].to(DEVICE)
                        target_lens = sup_data['txt_lens'][k].to(DEVICE)
                        sup_loss = task_model(inputs, targets, input_lens, target_lens)
                        task_optimizer.step(sup_loss)
                    qry_inputs = qry_data['vid'][k].to(DEVICE)
                    qry_targets = qry_data['txt'][k].to(DEVICE)
                    input_lens = qry_data['vid_lens'][k].to(DEVICE)
                    target_lens = qry_data['txt_lens'][k].to(DEVICE)
                    qry_loss = task_model(qry_inputs, qry_targets, input_lens, target_lens)
                    meta_losses.append(qry_loss.detach())
                    qry_loss.backward()
            meta_optimizer.step()
            meta_loss = sum(meta_losses) / len(meta_losses)
            # lr_scheduler.step()
            print("Iteration {}, Meta loss: {:.4f}".format(i + 1, meta_loss.data.item()), flush=True)
            if (i + 1) % 50 == 0:
                savename = 'iter_{}.pt'.format(i + 1)
                savedir = os.path.join('checkpoints', 'maml_grid')
                if not os.path.exists(savedir): os.makedirs(savedir)
                torch.save({'model': meta_model.state_dict()}, os.path.join(savedir, savename))


# meta testing过程：在新测试数据上，先用少量数据微调元模型，再进行评估
@torch.no_grad()
def mamal_test(model_path, data_path):
    model = MetaLearner().to(DEVICE)
    # checkpoint = torch.load(opt.load, map_location=lambda storage, loc: storage)
    checkpoint = torch.load(model_path, map_location='cpu')
    states = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(states)
    model.eval()
    spk_paths = [os.path.join(data_path, fn) for fn in os.listdir(data_path)]
    task = Task(spk_paths)
    pad = task.char_dict.index(PAD)
    bos = task.char_dict.index(BOS)
    eos = task.char_dict.index(EOS)
    preds = []
    refs = []
    for batch_data in task.seq_data_batch(32):
        vid_inp = batch_data['vid'].to(DEVICE)
        vid_lens = batch_data['vid_lens'].to(DEVICE)
        tgt_txt = batch_data['txt'].to(DEVICE)
        output = model.decoding(vid_inp, vid_lens, bos_id=bos, eos_id=eos)
        pred = []
        gold = []
        for out, tgt in zip(output, tgt_txt):
            pred.append(''.join([task.char_dict[i] for i in out.tolist() if i not in [pad, bos, eos]]))
            gold.append(''.join([task.char_dict[i] for i in tgt.tolist() if i not in [pad, bos, eos]]))
        print(pred, gold)
        preds.extend(pred)
        refs.extend(gold)
    test_wer, test_cer = wer(refs, preds), cer(refs, preds)
    print('JIWER wer: {:.4f}, cer: {:.4f}'.format(test_wer, test_cer))


if __name__ == '__main__':
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #dataset = MAMLDataset(r'E:\GRID\LIP_160_80\lip', batch_size=20)
    #with torch.backends.cudnn.flags(enabled=False):
    #maml_train(dataset, inner_lr=1e-4, meta_lr=1e-4, num_iters=10000, meta_batch=5)  # 内循环的lr更大
    #maml_train(dataset, inner_lr=1e-4, meta_lr=1e-4, num_iters=10000, meta_batch=4, model_path='checkpoints/grid/iter2_50.pt')  # 内循环的lr更大

    mamal_test('checkpoints/maml_grid/iter_500.pt', 'E:\GRID\LIP_160_80\lip\s20')

