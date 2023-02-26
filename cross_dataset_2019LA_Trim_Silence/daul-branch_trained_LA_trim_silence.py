import argparse
import json
import os
import shutil
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import ASVspoof2019
from resnet import setup_seed, ResNet, TypeClassifier
import torch.nn.functional as F
import eval_metrics as em

torch.set_default_tensor_type(torch.FloatTensor)

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)
    # Data folder prepare
    parser.add_argument("--access_type", type=str, default='LA')
    parser.add_argument("--data_path_cqt", type=str, default='../data/CQTFeatures_Silence/')
    parser.add_argument("--data_path_lfcc", type=str, default='../data/LFCCFeatures_Silence/')
    parser.add_argument("--data_protocol", type=str, help="protocol path",
                        default='../data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
    parser.add_argument("--out_fold", type=str, help="output folder",default='trained on LA_train_trim_silence/')

    # Dataset prepare
    parser.add_argument("--feat_len", type=int, help="features length", default=750)
    parser.add_argument('--padding', type=str, default='repeat')
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=32, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")
    parser.add_argument('--seed', type=int, help="random number seed", default=688)
    parser.add_argument('--lambda_', type=float, default=0.05, help="lambda for gradient reversal layer")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)

    if not os.path.exists(args.out_fold):
        os.makedirs(args.out_fold)
    else:
        shutil.rmtree(args.out_fold)
        os.mkdir(args.out_fold)
    if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
        os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
    else:
        shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
        os.mkdir(os.path.join(args.out_fold, 'checkpoint'))
    with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
        file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))
    with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
        file.write("Start recording training loss ...\n")
    with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
        file.write("Start recording validation loss ...\n")

    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args


def getFakeFeature(feature,label):
    f = []
    l = []
    for i in range(0,label.shape[0]):
        if label[i]!=20:
            l.append(label[i])
            f.append(feature[i])
    f = torch.stack(f)
    l = torch.stack(l)
    return f,l


def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)
    criterion = nn.CrossEntropyLoss()

    resnet_lfcc = ResNet(3, args.enc_dim, resnet_type='18', nclasses=2).to(args.device)
    resnet_cqt = ResNet(4, args.enc_dim, resnet_type='18', nclasses=2).to(args.device)
    classifier_lfcc = TypeClassifier(args.enc_dim, 6, args.lambda_, ADV=True).to(args.device)
    classifier_cqt = TypeClassifier(args.enc_dim, 6, args.lambda_, ADV=True).to(args.device)

    resnet_lfcc_optimizer = torch.optim.Adam(resnet_lfcc.parameters(),lr=args.lr, betas=(args.beta_1,args.beta_2),eps=args.eps, weight_decay=1e-4)
    resnet_cqt_optimizer = torch.optim.Adam(resnet_cqt.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=1e-4)
    classifier_lfcc_optimizer = torch.optim.Adam(classifier_lfcc.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2),eps=args.eps, weight_decay=1e-4)
    classifier_cqt_optimizer = torch.optim.Adam(classifier_cqt.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2),eps=args.eps, weight_decay=1e-4)

    trainset = ASVspoof2019(data_path_lfcc=args.data_path_lfcc,data_path_cqt=args.data_path_cqt,data_protocol=args.data_protocol,
                            access_type=args.access_type,data_part='train',feat_length=args.feat_len,padding=args.padding)
    validationset = ASVspoof2019(data_path_lfcc=args.data_path_lfcc,data_path_cqt=args.data_path_cqt,data_protocol='../data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
                            access_type=args.access_type,data_part='dev',feat_length=args.feat_len,padding=args.padding)
    trainDataLoader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=trainset.collate_fn)
    valDataLoader = DataLoader(validationset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=validationset.collate_fn)

    for epoch_num in range(args.num_epochs):
        print('\nEpoch: %d ' % (epoch_num + 1))

        resnet_lfcc.train()
        resnet_cqt.train()
        classifier_lfcc.train()
        classifier_cqt.train()

        epoch_loss = []
        epoch_lfcc_ftcloss = []
        epoch_lfcc_fcloss = []
        epoch_cqt_ftcloss = []
        epoch_cqt_fcloss = []

        for i, (lfcc,cqt,label,fakelabel) in enumerate(tqdm(trainDataLoader)):
            lfcc = lfcc.unsqueeze(1).float().to(args.device)
            cqt = cqt.unsqueeze(1).float().to(args.device)
            label = label.to(args.device)
            fakelabel = fakelabel.to(args.device)

            # get fake features and forgery type label
            feature_lfcc, out_lfcc = resnet_lfcc(lfcc)
            feature_fake_lfcc,fakelabel_lfcc = getFakeFeature(feature_lfcc,fakelabel)

            # calculate ftcloss
            typepred_lfcc = classifier_lfcc(feature_fake_lfcc)
            typeloss_lfcc = criterion(typepred_lfcc, fakelabel_lfcc)

            # optimize FTCM
            classifier_lfcc_optimizer.zero_grad()
            typeloss_lfcc.backward(retain_graph=True)
            classifier_lfcc_optimizer.step()

            # get new ftcloss
            type_pred_lfcc = classifier_lfcc(feature_fake_lfcc)
            ftcloss_lfcc = criterion(type_pred_lfcc, fakelabel_lfcc)

            # calculate fcloss
            fcloss_lfcc = criterion(out_lfcc,label)

            # cqt branch
            # get fake features and forgery type label
            feature_cqt, out_cqt = resnet_cqt(cqt)
            feature_fake_cqt, fakelabel_cqt = getFakeFeature(feature_cqt, fakelabel)

            # calculate ftcloss
            typepred_cqt = classifier_cqt(feature_fake_cqt)
            typeloss_cqt = criterion(typepred_cqt, fakelabel_cqt)

            # optimize FTCM
            classifier_cqt_optimizer.zero_grad()
            typeloss_cqt.backward(retain_graph=True)
            classifier_cqt_optimizer.step()

            # get new ftcloss
            type_pred_cqt = classifier_cqt(feature_fake_cqt)
            ftcloss_cqt = criterion(type_pred_cqt, fakelabel_cqt)

            # calculate fcloss
            fcloss_cqt = criterion(out_cqt, label)

            # LOSS
            loss = ftcloss_lfcc + fcloss_lfcc + ftcloss_cqt + fcloss_cqt
            epoch_loss.append(loss.item())

            epoch_lfcc_ftcloss.append(ftcloss_lfcc.item())
            epoch_lfcc_fcloss.append(fcloss_lfcc.item())
            epoch_cqt_ftcloss.append(ftcloss_cqt.item())
            epoch_cqt_fcloss.append(fcloss_cqt.item())

            # opyimize Feature Extraction Module and Forgery Classification Module
            resnet_lfcc_optimizer.zero_grad()
            resnet_cqt_optimizer.zero_grad()
            loss.backward()
            resnet_lfcc_optimizer.step()
            resnet_cqt_optimizer.step()


        with open(os.path.join(args.out_fold,'train_loss.log'),'a') as log:
            log.write(str(epoch_num+1) + '\t' +
                      'loss:' + str(np.nanmean(epoch_loss)) + '\t' +
                      'lfcc_fcloss:' + str(np.nanmean(epoch_lfcc_fcloss)) + '\t' +
                      'cqt_fcloss:' + str(np.nanmean(epoch_cqt_fcloss)) + '\t' +
                      'lfcc_ftcloss:' + str(np.nanmean(epoch_lfcc_ftcloss)) + '\t' +
                      'cqt_ftcloss:' + str(np.nanmean(epoch_cqt_ftcloss)) + '\t' +
                      '\n'
                      )

        resnet_lfcc.eval()
        resnet_cqt.eval()
        classifier_cqt.eval()
        classifier_lfcc.eval()

        with torch.no_grad():
            dev_loss = []
            label_list = []
            scores_list = []

            for i, (lfcc,cqt,label,_) in enumerate(tqdm(valDataLoader)):
                lfcc = lfcc.unsqueeze(1).float().to(args.device)
                cqt = cqt.unsqueeze(1).float().to(args.device)
                label = label.to(args.device)

                _, out_lfcc = resnet_lfcc(lfcc)
                fcloss_lfcc = criterion(out_lfcc, label)
                score_lfcc = F.softmax(out_lfcc, dim=1)[:, 0]

                _, out_cqt = resnet_cqt(cqt)
                fcloss_cqt = criterion(out_cqt, label)
                score_cqt = F.softmax(out_cqt, dim=1)[:, 0]

                score = torch.add(score_lfcc,score_cqt)
                score = torch.div(score,2)

                loss = fcloss_lfcc + fcloss_cqt
                dev_loss.append(loss.item())

                label_list.append(label)
                scores_list.append(score)

            scores = torch.cat(scores_list,0).data.cpu().numpy()
            labels = torch.cat(label_list,0).data.cpu().numpy()
            val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

            with open(os.path.join(args.out_fold, 'dev_loss.log'), 'a') as log:
                log.write(str(epoch_num + 1) + '\t' +
                          'loss:'+ str(np.nanmean(dev_loss)) + '\t' +
                          'val_eer:' + str(val_eer) + '\t' +
                          '\n')

        torch.save(resnet_lfcc, os.path.join(args.out_fold, 'checkpoint','anti-spoofing_lfcc_model_%d.pt' % (epoch_num + 1)))
        torch.save(resnet_cqt, os.path.join(args.out_fold, 'checkpoint','anti-spoofing_cqt_model_%d.pt' % (epoch_num + 1)))

    return resnet_lfcc


if __name__=='__main__':
    args = initParams()
    resnet = train(args)
    model = torch.load(os.path.join(args.out_fold,'anti-spoofing_lfcc_model.pt'))



