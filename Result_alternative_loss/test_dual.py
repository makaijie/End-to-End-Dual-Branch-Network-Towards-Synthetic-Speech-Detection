import argparse
import os
import shutil
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from data_utils import ASVspoof2019
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from tqdm import tqdm
import eval_metrics as em
import numpy as np

def test_model(lfcc_path,cqt_path, device):

    model_lfcc = torch.load(lfcc_path, map_location="cuda")
    model_cqt = torch.load(cqt_path, map_location="cuda")
    model_lfcc = model_lfcc.to(device)
    model_cqt = model_cqt.to(device)

    test_set = ASVspoof2019(data_path_lfcc='LFCCFeatures/',data_path_cqt='CQTFeatures/',data_protocol='LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt',
                            access_type='LA',data_part='eval',feat_length=750,padding='repeat')
    testDataLoader = DataLoader(test_set, batch_size=10,shuffle=False, num_workers=0,collate_fn=test_set.collate_fn)
    model_lfcc.eval()
    model_cqt.eval()


    with open(os.path.join('Result_alternative_loss', 'checkpoint_cm_score.txt'), 'w') as cm_score_file:
        for i, (filename,lfcc,cqt,labels,type) in enumerate(tqdm(testDataLoader)):
            lfcc = lfcc.unsqueeze(1).float().to(device)
            cqt = cqt.unsqueeze(1).float().to(device)
            labels = labels.to(device)

            _, score_lfcc = model_lfcc(lfcc)
            scores_lfcc = F.softmax(score_lfcc, dim=1)[:, 0]

            _, score_cqt = model_cqt(cqt)
            scores_cqt = F.softmax(score_cqt, dim=1)[:, 0]

            score = torch.add(scores_lfcc,scores_cqt)
            score = torch.div(score,2)

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (filename[j], type[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))
    eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join('Result_alternative_loss', 'checkpoint_cm_score.txt'),'')
    return eer_cm

def test(model_dir, device):

    lfcc_path = "anti-spoofing_lfcc_model_79.pt"
    cqt_path = "anti-spoofing_cqt_model_79.pt"

    lfcc_path = os.path.join(model_dir, lfcc_path)
    cqt_path = os.path.join(model_dir, cqt_path)

    test_model(lfcc_path,cqt_path, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model_dir', type=str, default="Result_alternative_loss/")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(args.model_dir, args.device)
