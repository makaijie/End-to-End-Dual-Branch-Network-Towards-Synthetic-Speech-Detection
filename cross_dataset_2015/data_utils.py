import random
import torch
import numpy as np
import os
import pickle
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from resnet import ResNet


class ASVspoof2015(Dataset):
    def __init__(self,data_path_lfcc,data_path_cqt,data_protocol,feat_length=750,padding='repeat'):
        self.data_path_lfcc = data_path_lfcc
        self.data_path_cqt = data_path_cqt
        self.data_protocol = data_protocol
        self.feat_len = feat_length
        self.padding = padding
        self.tag = {"human": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5,
                    "S6": 6, "S7": 7, "S8": 8, "S9": 9, "S10": 10}
        self.label = {"spoof": 1, "human": 0}

        with open(self.data_protocol, 'r') as f:
            self.audio_info = [info.strip().split() for info in f.readlines()]


    def __len__(self):
        return len(self.audio_info)


    def __getitem__(self, item):
        _,filename,_,labels = self.audio_info[item]

        with open(self.data_path_lfcc + filename + 'LFCC' + '.pkl', 'rb') as feature_handle:
            lfcc_data = pickle.load(feature_handle)
        cqt_path = self.data_path_cqt  + filename + '.npy'
        cqt_data = np.load(cqt_path)


        lfcc_data = torch.from_numpy(lfcc_data)
        lfcc_length = lfcc_data.shape[1]

        if lfcc_length > self.feat_len:
            start = np.random.randint(lfcc_length - self.feat_len)
            lfcc_data = lfcc_data[:, start: start + self.feat_len]
        if lfcc_length < self.feat_len:
            if self.padding == 'zero':
                lfcc_data = zero_padding(lfcc_data, self.feat_len)
            if self.padding == 'repeat':
                lfcc_data = repeat_padding(lfcc_data, self.feat_len)


        cqt_data = torch.from_numpy(cqt_data)
        cqt_length = cqt_data.shape[1]

        if cqt_length > self.feat_len:
            start = np.random.randint(cqt_length - self.feat_len)
            cqt_data = cqt_data[:, start: start + self.feat_len]
        if cqt_length < self.feat_len:
            if self.padding == 'zero':
                cqt_data = zero_padding(cqt_data, self.feat_len)
            if self.padding == 'repeat':
                cqt_data = repeat_padding(cqt_data, self.feat_len)
                

        return filename,lfcc_data,cqt_data,self.label[labels]

    def collate_fn(self, data):
        return default_collate(data)

def zero_padding(mat, feat_len):
    height, len = mat.shape
    pad_len = feat_len - len
    return torch.cat((mat, torch.zeros(height, pad_len, dtype=mat.dtype)), 1)

def repeat_padding(mat, feat_len):
    mul = int(np.ceil(feat_len / mat.shape[1]))
    mat = mat.repeat(1, mul)[:, :feat_len]
    return mat










































