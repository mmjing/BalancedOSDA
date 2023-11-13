from __future__ import print_function
import argparse
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os
import time
import gc
import torch
from pickle import dump, load
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import random
import sys

from utils import *
from utils2 import *
import mymodels
from data import get_mini_batches
from FeatureLoader import myPairedData
from basenet import *

def GetNowTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
def getTimestamp():
    return time.strftime("%m%d%H%M%S", time.localtime(time.time()))

parser = argparse.ArgumentParser(description='PyTorch Openset DA')
parser.add_argument('--dataset', type=str, default='image-clef')
parser.add_argument('--source', type=str, default='c')    
parser.add_argument('--target', type=str, default='b')                  
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=8)                    
parser.add_argument('--log_inter', type=int, default=10)
parser.add_argument('--mintail', type=int, default=2)                     
parser.add_argument('--decay', type=float, default=0.8) 
parser.add_argument('--end_iter', type=int, default=8000) 

parser.add_argument('--tailsize', type=float, default=0.02) 
parser.add_argument('--margin', type=float, default=3.0)
parser.add_argument('--loss_ca', type=float, default=1.0)                    
parser.add_argument('--loss_cnp', type=float, default=1.0)
parser.add_argument('--h_dim', type=int, default=256) 
parser.add_argument('--z_dim', type=int, default=128)
parser.add_argument('--lr_enc', type=float, default=4e-4)
parser.add_argument('--lr_dec', type=float, default=4e-4)
parser.add_argument('--lr_cls', type=float, default=1e-3)

args = parser.parse_args()
args.cuda = True
print(args)
print(GetNowTime())
print('Begin run!!!')
since = time.time()

root = './features/'

source_train_feat = np.load(root+args.source+'_source_train_feature.npy').astype('float32')
source_train_label = np.load(root+args.source+'_source_train_label.npy').astype('int')
target_train_feat = np.load(root+args.target+'_target_train_feature.npy').astype('float32')
target_train_label = np.load(root+args.target+'_target_train_label.npy').astype('int')

source_test_feat = np.load(root+args.source+'_source_test_feature.npy').astype('float32')
source_test_label = np.load(root+args.source+'_source_test_label.npy').astype('int')
target_test_feat = np.load(root+args.target+'_target_test_feature.npy').astype('float32')
target_test_label = np.load(root+args.target+'_target_test_label.npy').astype('int')

train_loader = myPairedData(source_train_feat,source_train_label,target_train_feat,target_train_label,args.batch_size)

batch_size = args.batch_size

data_test_target = get_mini_batches(target_test_feat, target_test_label, batch_size)
data_test_source = get_mini_batches(source_test_feat, source_test_label, batch_size)

use_gpu = torch.cuda.is_available()

num_class = 7

IN_DIM = 2048
H_DIM = args.h_dim
Z_DIM = args.z_dim

encoder = mymodels.Encoder(IN_DIM, args.h_dim, args.z_dim)
classifier = mymodels.LINEAR_LOGSOFTMAX(args.z_dim, num_class)

load_path = './saved_models.pth'
encoder, classifier, centroid_distance = load_model(encoder, classifier, load_path)

print('-----------')
if args.cuda:
    encoder.cuda()
    classifier.cuda()

encoder.eval()
classifier.eval()


def test(centroid_distance):   
    centroid_distance = centroid_distance
    per_class_feat_s = []
    for i in range(num_class-1):
        per_class_feat_s.append([])

    per_class_src_num = np.zeros((num_class-1)).astype(np.float32)
    per_class_src_correct = np.zeros((num_class-1)).astype(np.float32)
    for batch_idx, data in enumerate(data_test_source):
        img_s = torch.from_numpy(data[0]).cuda()
        label_s = torch.from_numpy(data[1])
        with torch.no_grad():
            mu_s, sigma_s = encoder(img_s)  
            # z_s = reparameterize(mu_s, sigma_s).rsample().squeeze() 
            z_s = mu_s.squeeze()
            out_s1 = classifier(z_s)
        
        feat_s = z_s.detach().cpu()
        pred = out_s1.detach().max(1)[1].cpu().numpy()

        batch_class = torch.unique(label_s).tolist()
        for s in batch_class:
            s_ind = np.where(label_s.data.cpu().numpy() == s)[0]
            if len(s_ind) == 0:
                continue
            per_class_src_num[s] += float(len(s_ind))
            correct_s_ind = np.where(pred[s_ind] == s)[0]
            per_class_src_correct[s] += float(len(correct_s_ind))

            correct_feat = feat_s[s_ind][correct_s_ind]
            if len(per_class_feat_s[s]) == 0:
                per_class_feat_s[s] = correct_feat
            else:
                per_class_feat_s[s] = torch.cat((per_class_feat_s[s],correct_feat),dim=0) 

    tailsizes = []
    for i in range(num_class-1):
        tailsizes.append( max(int(args.tailsize * per_class_src_correct[i]),args.mintail) )

    per_class_fit_num = []
    for s in range(num_class-1):
        if len(per_class_feat_s[s]) < 2 * tailsizes[s]:
            per_class_feat_s[s] = []
        per_class_fit_num.append(len(per_class_feat_s[s]))
    
    per_class_src_centroid = get_means(per_class_feat_s)       

    distances_to_z_means_correct_train = calc_distances_to_means(per_class_src_centroid, per_class_feat_s)

    del per_class_feat_s
    gc.collect()

    weibull_models, valid_weibull = fit_weibull_models(distances_to_z_means_correct_train, tailsizes)
    if not valid_weibull:
        print("Weibull fit is not valid")

    per_class_num = np.zeros((num_class))
    per_class_correct1 = np.zeros((num_class)).astype(np.float32)
    
    per_class_feat_t = []
    label_pred_class_list_t = [[],[]]
    for i in range(num_class-1):
        per_class_feat_t.append([])   

    for i in range(num_class):
        label_pred_class_list_t[0].append([])
        label_pred_class_list_t[1].append([])

    for batch_idx, data in enumerate(data_test_target):
        img_t = torch.from_numpy(data[0]).cuda()
        label_t = torch.from_numpy(data[1])
        with torch.no_grad():
            mu_t, sigma_t = encoder(img_t)   
            # z_t = reparameterize(mu_t, sigma_t).rsample().squeeze()                 
            z_t = mu_t.squeeze()
            out_t1 = classifier(z_t)

        pred = out_t1.detach().max(1)[1].cpu()
        batch_class = torch.unique(label_t).tolist()
        feat = z_t.detach().cpu()

        pred_class = torch.unique(pred).tolist()
        for t in pred_class:
            t_ind = np.where(pred.numpy() == t)[0]  
            if len(t_ind) == 0:
                continue            
            if len(label_pred_class_list_t[0][t])==0:
                label_pred_class_list_t[0][t] = label_t.numpy()[t_ind]
                label_pred_class_list_t[1][t] = pred.numpy()[t_ind]
            else:
                label_pred_class_list_t[0][t] = np.concatenate((label_pred_class_list_t[0][t],label_t.numpy()[t_ind]),axis=0)
                label_pred_class_list_t[1][t] = np.concatenate((label_pred_class_list_t[1][t],pred.numpy()[t_ind]),axis=0)

            if t == num_class-1:
                continue
            if len(per_class_feat_t[t]) == 0:
                per_class_feat_t[t] = feat[t_ind]
            else:
                per_class_feat_t[t] = torch.cat((per_class_feat_t[t],feat[t_ind]),dim=0) 


    distances_to_z_means_threshset = calc_distances_to_means(per_class_src_centroid, per_class_feat_t)    

    distances_t2s_corrected = correct_dist(distances_to_z_means_threshset,centroid_distance)

    outlier_probs_threshset = calc_outlier_probs(weibull_models, distances_t2s_corrected)
    
    open_OS_star_acc, open_unk_acc, open_H  = calc_mean_class_acc(outlier_probs_threshset,label_pred_class_list_t,num_class)          

    print('OS_star:{:.3f}, Unk:{:.3f}, H:{:.3f}'.format(open_OS_star_acc,open_unk_acc,open_H))

test(centroid_distance)
gc.collect()