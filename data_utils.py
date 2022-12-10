import random
import torch
import numpy as np
import os
import pickle
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader



class ASVspoof2019(Dataset):
    def __init__(self,data_path_lfcc,data_path_cqt,data_protocol,access_type='LA',data_part='train',feat_length=750,padding='repeat'):
        self.data_path_lfcc = data_path_lfcc
        self.data_path_cqt = data_path_cqt
        self.data_protocol = data_protocol
        self.access_type = access_type
        self.data_part = data_part
        self.feat_len = feat_length
        self.padding = padding
        self.tag = {"-": 20, "A01": 0, "A02": 1, "A03": 2, "A04": 3, "A05": 4, "A06": 5,
                    "A07": 7, "A08": 8, "A09": 9, "A10": 10, "A11": 11, "A12": 12, "A13": 13,
                    "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18, "A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}

        with open(self.data_protocol, 'r') as f:
            self.audio_info = [info.strip().split() for info in f.readlines()]

    def __len__(self):
        return len(self.audio_info)

    def __getitem__(self, item):
        speakerid, filename, _, fake_type, label = self.audio_info[item]
        if self.data_part=='train':
            with open(self.data_path_lfcc + 'train/' + filename + 'LFCC' + '.pkl', 'rb') as feature_handle:
                lfcc_data = pickle.load(feature_handle)
            cqt_path = self.data_path_cqt + 'train/' + filename + '.npy'
            cqt_data = np.load(cqt_path)
        if self.data_part == 'dev':
            with open(self.data_path_lfcc + 'dev/' + filename + 'LFCC' + '.pkl', 'rb') as feature_handle:
                lfcc_data = pickle.load(feature_handle)
            cqt_path = self.data_path_cqt + 'dev/' + filename + '.npy'
            cqt_data = np.load(cqt_path)
        if self.data_part == 'eval':
            with open(self.data_path_lfcc + 'eval/' + filename + 'LFCC' + '.pkl', 'rb') as feature_handle:
                lfcc_data = pickle.load(feature_handle)
            cqt_path = self.data_path_cqt + 'eval/' + filename + '.npy'
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
        return lfcc_data,cqt_data, self.label[label],self.tag[fake_type]

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







































