import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn

def ATCL_Loss(inputs, targets, centers, margin=1.0, eps=1e-6,reduction='mean',entropy_weights=None): 
    inputs = F.normalize(inputs)
    centers = F.normalize(centers)
    targets = targets.unsqueeze(1)
    dist = torch.acos(torch.clamp(torch.matmul(inputs,centers.transpose(0,1)), -1.+eps, 1-eps))

    dist_ap = torch.gather(dist,1,targets)
    if dist_ap.shape[0] == 1:
        dist_ap = dist_ap.squeeze().unsqueeze(0)
    else:
        dist_ap = dist_ap.squeeze()
        
    dist_onehot = torch.zeros_like(dist).scatter_(1, targets, 1)
    index_an = (torch.ones_like(dist) - dist_onehot).byte()

    dist_multi_an = dist[index_an].reshape(dist.shape[0],-1)

    dist_an = dist_multi_an.min(1)[0]

    y = torch.ones(inputs.shape[0]).cuda()

    loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin, reduction=reduction)
    if reduction == 'none':
        loss = torch.mul(loss, entropy_weights).mean()
    return loss

def ATCL_dist(inputs, targets,  centers, eps=1e-6): 
    inputs = F.normalize(inputs)
    centers = F.normalize(centers.float())
    targets = targets.unsqueeze(1)
    dist = torch.acos(torch.clamp(torch.matmul(inputs,centers.transpose(0,1)), -1.+eps, 1-eps))
    dist_ap = torch.gather(dist,1,targets)
    return dist_ap

def angular_dist(inputs, centers, eps=1e-6): 
    # num_class = len(inputs)
    inputs = F.normalize(inputs)
    centers = F.normalize(centers)
    dist = torch.acos(torch.clamp(torch.matmul(inputs,centers.transpose(0,1)), -1.+eps, 1-eps))
    return dist


def targets_unknown_loss(target_unk, src_centroids, margins=1.0, eps=1e-6,reduction='mean',entropy_weights=None):
    target_unk = F.normalize(target_unk)
    src_centroids = F.normalize(src_centroids)
    if target_unk.size(0) < 1:
        return torch.tensor(0)
    else:             
        dist = torch.acos(torch.clamp(torch.matmul(target_unk, src_centroids.transpose(0,1)), -1.+eps, 1-eps))
        margin_multi = torch.tensor(margins).expand_as(dist).cuda()
        dist_hinge = torch.clamp(margin_multi - dist, min=0.0)  
        if reduction == 'mean':      
            loss = torch.mean(dist_hinge)
        else:
            loss = torch.mul(dist_hinge,entropy_weights.unsqueeze(1).expand_as(dist_hinge)).mean()
        return loss