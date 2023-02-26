import argparse
import os
import shutil
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from data_utils import ASVspoof2015
from tqdm import tqdm
import numpy as np
import eval_metrics as em


def test_model(lfcc_path,cqt_path, device):
    model_lfcc = torch.load(lfcc_path, map_location="cuda")
    model_cqt = torch.load(cqt_path, map_location="cuda")
    model_lfcc = model_lfcc.to(device)
    model_cqt = model_cqt.to(device)

    test_set = ASVspoof2015(data_path_lfcc='../data/ASVspoof2015/LFCC_2015_pkl/',data_path_cqt='../data/ASVspoof2015/CQTFeatures_2015/',
                              data_protocol='../data/ASVspoof2015/CM_protocol/cm_evaluation.ndx', feat_length=750,padding='repeat')
    testDataLoader = DataLoader(test_set, batch_size=20,shuffle=False, num_workers=0,collate_fn=test_set.collate_fn)
    model_lfcc.eval()
    model_cqt.eval()

    score_loader, idx_loader = [], []

    score_txt_path = '2015_score.txt'
    with open(os.path.join('test', score_txt_path), 'w') as cm_score_file:
        for i, (filename, lfcc, cqt,labels) in enumerate(tqdm(testDataLoader)):
            lfcc = lfcc.unsqueeze(1).float().to(device)
            cqt = cqt.unsqueeze(1).float().to(device)
            labels = labels.to(device)

            _, score_lfcc = model_lfcc(lfcc)
            scores_lfcc = F.softmax(score_lfcc, dim=1)[:, 0]

            _, score_cqt = model_cqt(cqt)
            scores_cqt = F.softmax(score_cqt, dim=1)[:, 0]

            score = torch.add(scores_lfcc,scores_cqt)
            score = torch.div(score,2)

            for j in range(score.size(0)):
                cm_score_file.write(
                    '%s %s\n' % (filename[j], score[j].item()))

            score_loader.append(score.detach().cpu())
            idx_loader.append(labels.detach().cpu())

    scores = torch.cat(score_loader, 0).data.cpu().numpy()
    labels = torch.cat(idx_loader, 0).data.cpu().numpy()
    eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
    other_eer = em.compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
    eer = min(eer, other_eer)
    print(eer)


    return eer


def test(model_dir, device):

    lfcc_path = "anti-spoofing_lfcc_model.pt"
    cqt_path = "anti-spoofing_cqt_model.pt"
    lfcc_path = os.path.join(model_dir, lfcc_path)
    cqt_path = os.path.join(model_dir, cqt_path)
    test_model(lfcc_path, cqt_path, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model_dir', type=str, default="test/")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(args.model_dir, args.device)

