#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import sys
import copy
import argparse
from PIL import Image
from torchvision.models import wide_resnet50_2

try:
    import cPickle as pickle
except:
    import pickle
import math
import logging

from utils_incremental.compute_features import compute_features
from utils_incremental.incremental_train_and_eval_Graph import incremental_train_and_eval_Graph
import copy
import dataload

os.environ["CUDA_VISIBLE_DEVICES"] = "4,1,7,9"
######### Modifiable Settings ##########
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--nb_cl_fg', default=50, type=int, \
                    help='the number of classes in first group')
parser.add_argument('--nb_cl', default=3, type=int, \
                    help='Classes per group')
parser.add_argument('--nb_protos', default=20, type=int, \
                    help='Number of prototypes per class at the end')
parser.add_argument('--nb_runs', default=1, type=int, \
                    help='Number of runs (random ordering of classes at each run)')
parser.add_argument('--ckp_prefix', default=os.path.basename(sys.argv[0])[:-3], type=str, \
                    help='Checkpoint prefix')
parser.add_argument('--T', default=2, type=float, \
                    help='Temporature for distialltion')
parser.add_argument('--beta', default=0.25, type=float, \
                    help='Beta for distialltion')
parser.add_argument('--resume', action='store_true', \
                    help='resume from checkpoint')
parser.add_argument('--fix_budget', action='store_true', \
                    help='fix budget')
########################################
parser.add_argument('--mimic_score', action='store_true', \
                    help='To mimic scores for cosine embedding')
parser.add_argument('--lw_ms', default=1, type=float, \
                    help='loss weight for mimicking score')
########################################
# improved class incremental learning
parser.add_argument('--rs_ratio', default=0, type=float, \
                    help='The ratio for resample')
parser.add_argument('--imprint_weights', action='store_true', \
                    help='Imprint the weights for novel classes')
parser.add_argument('--less_forget', action='store_true', default=True, \
                    help='Less forgetful')
parser.add_argument('--lamda', default=5, type=float, \
                    help='Lamda for LF')
parser.add_argument('--adapt_lamda', action='store_true', \
                    help='Adaptively change lamda')
parser.add_argument('--mr_loss', action='store_true', default=True, \
                    help='Margin ranking loss v1')
parser.add_argument('--amr_loss', action='store_true', \
                    help='Margin ranking loss v2')
parser.add_argument('--dist', default=0.5, type=float, \
                    help='Dist for MarginRankingLoss')
parser.add_argument('--K', default=2, type=int, \
                    help='K for MarginRankingLoss')
parser.add_argument('--lw_mr', default=1, type=float, \
                    help='loss weight for margin ranking loss')
########################################
parser.add_argument('--random_seed', default=1993, type=int, \
                    help='random seed')
parser.add_argument('--log_dir', default='./log', type=str, \
                    help='log dir')
parser.add_argument('--graph_lambda', default=1, type=float)
parser.add_argument('--ref_nn', default=5, type=int)
parser.add_argument('--cls_weight', default=1, type=float)
################################################ finetune sets ########################
parser.add_argument("--data_path", type=str, default="./BTADFT/")
parser.add_argument('--nb_protos_cl', default=5, type=int)
parser.add_argument('--pre_class_num', default=1000, type=int)
parser.add_argument('--img_num_class', default=50, type=int)

args = parser.parse_args()

# logger
########################################
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

log_save_dir = os.path.join(args.log_dir, args.ckp_prefix + '.txt')
fh = logging.FileHandler(log_save_dir)
fh.setLevel(logging.INFO)
logger.addHandler(fh)

# h1 = logging.StreamHandler(sys.stdout)
# logger.addHandler(h1)
#################### pearson ####################
# assert(args.nb_cl_fg % args.nb_cl == 0)
# assert(args.nb_cl_fg >= args.nb_cl)

train_batch_size = 32  # Batch size for train
test_batch_size = 100  # Batch size for test
eval_batch_size = 128  # Batch size for eval
base_lr = [0.1, 0.000005]  # Initial learning rate  (pearson)
fc_lr = 0.0  # FC layer learning rate
epochs = 50  # Epochs
lr_strat = []  # Epochs where learning rate gets decreased
lr_factor = 0.1  # Learning rate decrease factor
custom_weight_decay = 5e-4  # Weight Decay
custom_momentum = 0.9  # Momentum
np.random.seed(args.random_seed)  # Fix the random seed
logger.info(args)
########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

if not os.path.exists('./checkpoint/{}/anchor_x.npy'.format(args.ckp_prefix)):
    # trainset = dataload.ft_dataset(args.data_path, empty=False, transform=data_transform)
    anchorset = dataload.pre_dataset('/dataset/imagenet/ILSVRC2012_img_val', empty=True, transform=data_transform)
    logger.info("anchorset prepared")
    evalset = dataload.pre_dataset('/dataset/imagenet/ILSVRC2012_img_val', empty=False, transform=data_transform)
    logger.info("evalset prepared")

    # prepare list
    graph_herding = np.zeros((args.pre_class_num, args.img_num_class), np.float32)
    order = np.arange(args.pre_class_num)
    order_list = list(order)

    prototypes = np.zeros(
        (args.pre_class_num, args.img_num_class, evalset.data.shape[1], evalset.data.shape[2], evalset.data.shape[3]))
    for orde in range(args.pre_class_num):
        prototypes[orde, :, :, :, :] = evalset.data[np.where(evalset.targets == orde)]
    Y_train_cumuls = []
    X_train_cumuls = []  # train images and anchor iamges

    # prepare model
    tg_model = wide_resnet50_2(pretrained=True, progress=True)
    tg_model.to(device)

    in_features = tg_model.fc.in_features
    out_features = tg_model.fc.out_features
    logger.info("in_features: {} out_features: {}".format(in_features, out_features))
    ref_model = None

    # calculate center of the pre-dataset
    tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
    num_features = tg_model.fc.in_features
    logger.info('Updating graph')

    # herding sample
    nb_protos_cl = args.nb_protos_cl
    for iter_dico in range(0, args.pre_class_num):
        evalset.data = prototypes[iter_dico].astype('uint8')
        evalset.targets = np.zeros(evalset.data.shape[0])  # zero labels
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                 shuffle=False, num_workers=2)
        num_samples = evalset.data.shape[0]
        mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)
        mapped_prototypes = mapped_prototypes.numpy()
        D = mapped_prototypes.T
        D = D / np.linalg.norm(D, axis=0)

        mu = np.mean(D, axis=1)
        graph_herding[iter_dico, :] = graph_herding[iter_dico, :] * 0
        w_t = mu
        iter_herding = 0
        iter_herding_eff = 0
        # if iter_dico == 100:
        #     pdb.set_trace()
        while not (np.sum(graph_herding[iter_dico, :] != 0) == min(nb_protos_cl, 500)) and iter_herding_eff < 1000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if graph_herding[iter_dico, ind_max] == 0:
                graph_herding[iter_dico, ind_max] = 1 + iter_herding
                iter_herding += 1
            w_t = w_t + mu - D[:, ind_max]

    graph_herding = (graph_herding > 0) * (graph_herding < nb_protos_cl + 1) * 1.

    X_protoset_cumuls = []
    Y_protoset_cumuls = []

    # register the anchor img
    for cl in range(0, args.pre_class_num):
        alph = graph_herding[cl, :]
        X_protoset_cumuls.append(prototypes[cl, np.where(alph == 1)[
            0]])  # prototypes[classes, samples, w, h, c]
        Y_protoset_cumuls.append(order[cl] * np.ones(len(np.where(alph == 1)[0])))
    X_protoset = np.concatenate(X_protoset_cumuls)
    Y_protoset = np.concatenate(Y_protoset_cumuls)

    # Graphing and Save the graph of all anchors
    #####################################################
    anchor_X = X_protoset
    anchor_Y = Y_protoset
    # map_anchor_Y = np.array([order_list.index(i) for i in anchor_Y])
    ckp_dir = './checkpoint/{}/'.format(args.ckp_prefix)
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)

    anchorset.data = anchor_X.astype('uint8')
    anchorset.targets = anchor_Y
    np.save('./checkpoint/{}/anchor_x.npy'.format(args.ckp_prefix), anchorset.data)
    np.save('./checkpoint/{}/anchor_y.npy'.format(args.ckp_prefix), anchorset.targets)
else:
    anchorset = dataload.pre_dataset('/dataset/imagenet/ILSVRC2012_img_val', empty=True, transform=data_transform)
    logger.info("anchorset prepared")
    anchorset.data = np.load('./checkpoint/{}/anchor_x.npy'.format(args.ckp_prefix))
    X_protoset = anchorset.data
    anchorset.targets = np.load('./checkpoint/{}/anchor_y.npy'.format(args.ckp_prefix))
    Y_protoset = anchorset.targets
    tg_model = wide_resnet50_2(pretrained=True, progress=True)
    tg_model.to(device)
    num_features = tg_model.fc.in_features

# prepare for finetune
inc = 1
ref_model = copy.deepcopy(tg_model)
in_features = tg_model.fc.in_features
out_features = tg_model.fc.out_features
logger.info("in_features: {} out_features: {}".format(in_features, out_features))
new_fc = modified_linear.SplitCosineLinear(in_features, out_features, args.nb_cl, sigma=False)
new_fc.fc1.weight.data = tg_model.fc.weight.data
# new_fc.sigma.data = tg_model.fc.sigma.data
tg_model.fc = new_fc
lamda_mult = out_features * 1.0 / args.nb_cl

# prepare fintune dataset
trainset = dataload.ft_dataset(args.data_path, empty=False, transform=data_transform)
X_train = trainset.data
Y_train = trainset.targets

X_train = np.concatenate((X_train, X_protoset), axis=0)
Y_train = np.concatenate((Y_train, Y_protoset))

trainset.data = X_train.astype('uint8')
# map_Y_train = np.array([order_list.index(i) for i in Y_train])
trainset.targets = Y_train

trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

# finetune
cur_lamda = args.lamda
ckp_name = './checkpoint/{}/epoch_{}_model5-6_ablation.pth'.format(args.ckp_prefix, epochs)
ckp_name = './checkpoint/{}/epoch_{}_model5-6_BTAD.pth'.format(args.ckp_prefix, epochs)

ignored_params = list(map(id, tg_model.fc.fc1.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params,
                     tg_model.parameters())
tg_params = [{'params': base_params, 'lr': base_lr[inc], 'weight_decay': custom_weight_decay},
             {'params': tg_model.fc.fc1.parameters(), 'lr': fc_lr, 'weight_decay': 0}]

# tg_model = tg_model.to(device)
# ref_model = ref_model.to(device)
tg_optimizer = optim.SGD(tg_params, lr=base_lr[inc], momentum=custom_momentum, weight_decay=custom_weight_decay)
tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)
###############################
logger.info("incremental_train_and_eval_Graph")
for layer in tg_model.modules():
    if isinstance(layer, nn.BatchNorm2d):
        layer.eval()

for layer in ref_model.modules():
    if isinstance(layer, nn.BatchNorm2d):
        layer.eval()

tg_model = incremental_train_and_eval_Graph(args, epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler,
                                            trainloader,
                                            1, 0,
                                            cur_lamda,
                                            args.dist, args.K, args.lw_mr, logger=logger, ckp_name=ckp_name,
                                            anchorset=anchorset, ckp_prefix=args.ckp_prefix,
                                            num_features=num_features)
tg_model.fc = nn.Sequential()
torch.save(tg_model, ckp_name)
