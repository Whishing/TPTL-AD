import torch
from scipy.stats import pearsonr
from torch import nn
from torch.nn import functional as F


def pearson_loss(ref_norm_feat, cur_norm_feat):
    # ref_norm_feat = F.normalize(ref_norm_feat, dim=1)
    # cur_norm_feat = F.normalize(cur_norm_feat, dim=1)
    ref_rank = torch.mm(ref_norm_feat.detach(), (ref_norm_feat.detach().transpose(1, 0)))
    cur_rank = torch.mm(cur_norm_feat, (cur_norm_feat.transpose(1, 0)))
    x, y = ref_rank.shape

    mref = torch.mean(ref_rank, 1)
    mcur = torch.mean(cur_rank, 1)

    refm = ref_rank - mref.repeat(y).reshape(y, x).transpose(1, 0)
    curm = cur_rank - mcur.repeat(y).reshape(y, x).transpose(1, 0)
    refm = refm.to(curm.device)
    r_num = torch.sum(refm * curm, 1)
    r_den = torch.sqrt(torch.sum(torch.pow(refm, 2), 1) * torch.sum(torch.pow(curm, 2), 1))
    r = 1 - (r_num / r_den)
    cor = torch.mean(r)
    return cor
