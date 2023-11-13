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
parser.add_argument('--margin', type=float, default=2.5)
parser.add_argument('--loss_ca', type=float, default=1.0)                    
parser.add_argument('--loss_cnp', type=float, default=1.0)
parser.add_argument('--h_dim', type=int, default=256) 
parser.add_argument('--z_dim', type=int, default=128)
parser.add_argument('--lr_enc', type=float, default=4e-4)
parser.add_argument('--lr_dec', type=float, default=4e-4)
parser.add_argument('--lr_cls', type=float, default=1e-3)

args = parser.parse_args()
args.cuda = True
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

print(args)

num_class = 7

IN_DIM = 2048
H_DIM = args.h_dim
Z_DIM = args.z_dim

encoder = mymodels.Encoder(IN_DIM, args.h_dim, args.z_dim)
decoder = mymodels.Decoder(args.z_dim, args.h_dim, IN_DIM)
classifier = mymodels.LINEAR_LOGSOFTMAX(args.z_dim, num_class)

optimizer_enc = optim.Adam(encoder.parameters(), lr = args.lr_enc) #
optimizer_dec = optim.Adam(decoder.parameters(), lr = args.lr_dec)
optimizer_cls = optim.Adam(classifier.parameters(), lr = args.lr_cls)
        
enc_scheduler = StepLR(optimizer_enc, step_size=10000, gamma=0.5)
dec_scheduler = StepLR(optimizer_dec, step_size=10000, gamma=0.5)
classifier_scheduler = StepLR(optimizer_cls, step_size=10000, gamma=0.5)

print('-----------')
if args.cuda:
    encoder.cuda()
    decoder.cuda()
    classifier.cuda()

def train(num_epoch):
    criterion = nn.CrossEntropyLoss().cuda()
    L1loss = nn.L1Loss()

    i = 0
    print('train start!')
    print(GetNowTime())
    since = time.time()

    open_best_OS_star_acc = 0
    open_best_unk_acc = 0
    open_best_H_acc = 0
    open_best_ep = -1


    src_class_centroid = torch.zeros(num_class-1, Z_DIM).cuda()
    tgt_class_centroid = torch.zeros(num_class-1, Z_DIM).cuda()  
    centroid_discrepancy = torch.zeros(num_class-1, Z_DIM).cuda()  

    encoder.train()
    decoder.train()
    classifier.train()
    for batch_idx, data in enumerate(train_loader):
        i += 1
        if batch_idx % 100 == 0:
            print('Iter {:d}'.format(batch_idx))             
        img_s = data['S']
        label_s = data['S_label']
        img_t = data['T']
        img_s = torch.from_numpy(img_s).cuda()
        label_s = torch.from_numpy(label_s).cuda()
        img_t = torch.from_numpy(img_t).cuda()

        if len(img_t) < batch_size:
            break
        if len(img_s) < batch_size:
            break

        encoder.zero_grad()
        decoder.zero_grad()
        classifier.zero_grad()

        mu_s, sigma_s = encoder(img_s)
        z_1 = reparameterize(mu_s, sigma_s, distribution='vmf')
        mu_t, sigma_t = encoder(img_t)
        z_2 = reparameterize(mu_t, sigma_t, distribution='vmf')
        
        z_s = z_1.rsample()
        z_t = z_2.rsample()
        
        out_s = classifier(z_s)
        out_t = classifier(z_t)

        softmax_s = F.softmax(out_s,dim=1) 
        softmax_t = F.softmax(out_t,dim=1)  
        cls_loss = criterion(out_s, label_s) 

        entropy_loss = -torch.mean(torch.log(torch.mean(softmax_t,0)+1e-6))
        cls_loss += 0.01*entropy_loss

        target_funk = torch.FloatTensor(z_t.shape[0], 2).fill_(0.5).cuda()
        out_tr = classifier(z_t, reverse=True, iter_num = 0 )
        out_tr = F.softmax(out_tr,dim=1)
        prob1 = torch.sum(out_tr[:, :num_class - 1], 1).view(-1, 1)
        prob2 = out_tr[:, num_class - 1].contiguous().view(-1, 1)
        prob = torch.cat((prob1, prob2), 1)
        loss_t = bce_loss(prob, target_funk)
        cls_loss += loss_t

        img_s2s = decoder(z_s)
        img_t2t = decoder(z_t)
        recon_loss_s2s = L1loss(img_s2s, img_s)
        recon_loss_t2t = L1loss(img_t2t, img_t)

        loss_KL = torch.distributions.kl.kl_divergence(z_1, z_2).mean()

        vae_loss = recon_loss_s2s + recon_loss_t2t + loss_KL

        loss = vae_loss + args.loss_cnp * cls_loss

        labels_t_pseudo = torch.max(softmax_t, 1)[1].cuda()
        labels_t_known_pseudo = labels_t_pseudo[labels_t_pseudo<num_class-1]
        feat_t_known = z_t[labels_t_pseudo<num_class-1]
        labels_t_unknown_pseudo = labels_t_pseudo[labels_t_pseudo==num_class-1]
        feat_t_unknown = z_t[labels_t_pseudo==num_class-1]
        target_known_batch = feat_t_known.shape[0]

        tgt_entropy = calc_entropy(out_t)
        tgt_known_entropy = tgt_entropy[labels_t_pseudo<num_class-1]
        tgt_unknown_entropy = tgt_entropy[labels_t_pseudo==num_class-1]
        weights_known = 1.0 + torch.exp(-tgt_known_entropy)
        weights_unknown = 1.0 + torch.exp(-tgt_unknown_entropy)
        all_weights = 1.0 + torch.exp(-tgt_entropy)


        len_weights_known = weights_known.sum()
        len_weights_unknown = weights_unknown.sum()      
        len_all_weights = all_weights.sum()   

        weights_known_expand = weights_known.unsqueeze(1).expand_as(feat_t_known)
        weights_all_expand = all_weights.unsqueeze(1).expand_as(z_t)
        feat_t_weighted = torch.mul(z_t,weights_all_expand)

        decay = args.decay
        update_rate = 1-decay  

        tgt_current_sum = unsorted_segment_sum(feat_t_weighted,labels_t_pseudo,num_class)            
        tgt_current_num = unsorted_segment_sum(weights_all_expand,labels_t_pseudo,num_class )
        tgt_positive_num = torch.max(tgt_current_num, torch.ones_like(tgt_current_num).cuda())
        tgt_current_centroid = torch.div(tgt_current_sum,tgt_positive_num)
        tgt_current_centroid = tgt_current_centroid[:num_class-1,:]

        tgt_adaptive_update_rate = torch.mul((update_rate * num_class/batch_size), tgt_current_num)[:num_class-1,:]
        tgt_class_centroid = torch.mul((1-tgt_adaptive_update_rate), tgt_class_centroid) + torch.mul(tgt_adaptive_update_rate , tgt_current_centroid)
            
        src_current_sum = unsorted_segment_sum(z_s,label_s,num_class-1)  
        src_ones = torch.ones(z_s.shape).cuda()
        src_current_num = unsorted_segment_sum(src_ones,label_s,num_class-1 )
        src_positive_num = torch.max(src_current_num, torch.ones_like(src_current_num).cuda())
        src_current_centroid = torch.div(src_current_sum,src_positive_num)

        src_adaptive_update_rate = torch.mul((update_rate * num_class/batch_size), src_current_sum)
        src_class_centroid = torch.mul((1-src_adaptive_update_rate), src_class_centroid) + torch.mul(src_adaptive_update_rate , src_current_centroid)

        loss_atcl_s2s = ATCL_Loss(z_s,label_s,src_class_centroid, args.margin)
        tgt_centroid_label = torch.tensor(list(range(num_class-1))).cuda()
        loss_atcl_t2s2 = ATCL_Loss(tgt_class_centroid,tgt_centroid_label,src_class_centroid, args.margin)   
        ca_loss = loss_atcl_s2s + loss_atcl_t2s2

        if feat_t_known.shape[0] > 0:
            loss_atcl_t2s = ATCL_Loss(feat_t_known,labels_t_known_pseudo,src_class_centroid,args.margin,reduction='none',entropy_weights=weights_known)
            ca_loss += 0.02 * loss_atcl_t2s
        if feat_t_unknown.shape[0] > 0:
            loss_unk = targets_unknown_loss(feat_t_unknown, src_class_centroid, args.margin,reduction='none',entropy_weights=weights_unknown)
            ca_loss += loss_unk

        loss += args.loss_ca * ca_loss

        loss.backward()     
        optimizer_enc.step()
        optimizer_dec.step()
        optimizer_cls.step()       

        src_class_centroid = src_class_centroid.detach()
        tgt_class_centroid = tgt_class_centroid.detach()        

        if  batch_idx % args.log_inter == (args.log_inter-1):
            encoder.eval()
            classifier.eval()
            tgt_centroid_label = torch.tensor(list(range(num_class-1))).cuda()
            centroid_distance = ATCL_dist(tgt_class_centroid,tgt_centroid_label,src_class_centroid).detach().cpu()                
            evt_results_dict = test(centroid_distance)
            open_H_acc = evt_results_dict['H_acc']
            if open_H_acc > open_best_H_acc:
                open_best_H_acc = open_H_acc
                open_best_ep = batch_idx
                open_best_OS_star_acc = evt_results_dict['OS_star_acc']
                open_best_unk_acc = evt_results_dict['Unk_acc']

            print('OS_star:{:.3f}, Unk:{:.3f}, H:{:.3f}, EP:{:d}'.format(open_best_OS_star_acc,open_best_unk_acc,open_best_H_acc,open_best_ep))               
            print('------------------------------------------------------------------------------------')
            sys.stdout.flush()            
            encoder.train()
            classifier.train()

        if batch_idx >= args.end_iter:
            break


    print('Final_OS_star:{:.3f}, Unk:{:.3f}, H:{:.3f}, EP:{:d}'.format(open_best_OS_star_acc,open_best_unk_acc,open_best_H_acc,open_best_ep))        
    time_elapsed = time.time() - since
    print('Time Elapsed: {}'.format(time_elapsed))
    print(GetNowTime())
    print('*******************************************************************************************\n')


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

    evt_results_dict = {
        'OS_star_acc':open_OS_star_acc, 'Unk_acc':open_unk_acc, 'H_acc': open_H
    }

    return evt_results_dict

train(args.end_iter + 1)
gc.collect()