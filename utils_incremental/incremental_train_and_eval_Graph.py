#!/usr/bin/env python
# coding=utf-8
import pdb
import sys

import torch

sys.path.extend(['./', '../'])
from torch.nn import functional as F
from utils_incremental.compute_features import compute_features
import copy
from utils_incremental.pearson_loss import pearson_los
from tqdm import tqdm

cur_features = []
ref_features = []
old_scores = []
new_scores = []


def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]


def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]


def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs


def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs


def incremental_train_and_eval_Graph(args, epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
                                     trainloader, \
                                     iteration, start_iteration, \
                                     lamda, \
                                     dist, K, lw_mr, \
                                     fix_bn=True, weight_per_class=None, device=None, logger=None, ckp_name=None,
                                     anchorset=None, ckp_prefix='default',
                                     num_features=64, ehg_bmu=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if iteration > start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features
        handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
        handle_cur_features = tg_model.fc.register_forward_hook(get_cur_features)
        anchorloader = torch.utils.data.DataLoader(anchorset, batch_size=64, shuffle=True, num_workers=2)
        anchoriter = iter(anchorloader)

    best_acc = 0
    # 多卡训练
    tg_model = nn.DataParallel(tg_model).to(device)
    ref_model = nn.DataParallel(ref_model).to(device)
    
    for epoch in range(epochs):

        tg_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    # m.track_running_stats = False
                    m.eval()

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        train_lossmm = 0
        old_correct = 0
        new_correct = 0
        old_total = 0
        new_total = 0
        correct = 0
        total = 0
        tg_lr_scheduler.step()
        logger.info('\nEpoch: %d, LR: ' % epoch)
        logger.info(tg_lr_scheduler.get_lr())

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            tg_optimizer.zero_grad()
            if iteration == start_iteration:
                outputs = tg_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            else:
                try:
                    inputsANC, targetsANC = anchoriter.next()
                except:
                    anchoriter = iter(anchorloader)
                    inputsANC, targetsANC = anchoriter.next()
                anchor_inputs = inputsANC.to(device)
                anchor_targets = targetsANC.to(device)
                # ref_anchor_inputs = copy.deepcopy(inputsANC).cuda(device=1)
                #################################################
                outputs = tg_model(anchor_inputs)
                ref_outputs = ref_model(anchor_inputs)

                graph_lambda = args.graph_lambda

                loss4 = pearson_loss(ref_features, cur_features) * graph_lambda

                #################################################

                #################################################
                # Org Loss1: fix anchor + novel
                outputs = tg_model(inputs)
                #################################################
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets.long()) * args.cls_weight
                #################################################
 
                #################################################
                loss2 = loss2.to(loss4.device)
                loss = loss2 + loss4

            loss.backward()
            tg_optimizer.step()

            train_loss += loss.item()
            if iteration > start_iteration:
                # train_loss1 += loss1.item() / lamda
                train_loss2 += loss2.item()
                # train_loss3 += loss3.item() / lw_mr
                train_loss4 += loss4.item()
                # train_lossmm += loss_minmax.item() / lw_mr
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            old_targets = targets.clone().detach().double()
            new_targets = targets.clone().detach().double()
            tmp_target = (torch.ones(targets.size(0), dtype=torch.float64) * -1).cuda()

            old_targets = torch.where(old_targets >= 1000, tmp_target, old_targets)
            new_targets = torch.where(new_targets < 1000, tmp_target, new_targets)
            old_correct += predicted.eq(old_targets).sum().item()
            new_correct += predicted.eq(new_targets).sum().item()
            old_total += len(targets[targets < 1000])
            new_total += len(targets[targets >= 1000])

        if iteration == start_iteration:
            logger.info('Train set: {}, Train Loss: {:.4f} Acc: {:.4f}'.format( \
                len(trainloader), train_loss / (batch_idx + 1), 100. * correct / total))
        else:
            logger.info(
                'Train set: {}, Train Loss1: {:.4f}, Train Loss2: {:.4f}, Train Loss3: {:.4f}, Train Loss4: {:.4f}, Train Loss_MinMax: {:.4f}'
                '\n Train Loss: {:.4f} Acc: {:.4f}'.format(len(trainloader), \
                                                           train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                                                           train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1),
                                                           train_lossmm / (batch_idx + 1),
                                                           train_loss / (batch_idx + 1), 100. * correct / total))
            print("{}".format(epoch))
            print("old acc  {:.3f}\t new acc  {:.3f}".format(100*old_correct / (old_total+1), 100*new_correct / (new_total+1)))
            print(
                'Train set: {}, Train Loss1: {:.4f}, Train Loss2: {:.4f}, Train Loss3: {:.4f}, Train Loss4: {:.4f}, Train Loss_MinMax: {:.4f}'
                '\n Train Loss: {:.4f} Acc: {:.4f}'.format(len(trainloader), \
                                                           train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                                                           train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1),
                                                           train_lossmm / (batch_idx + 1),
                                                           train_loss / (batch_idx + 1), 100. * correct / total))


    if iteration > start_iteration:
        logger.info("Removing register_forward_hook")
        handle_ref_features.remove()
        handle_cur_features.remove()
        # handle_old_scores_bs.remove()
        # handle_new_scores_bs.remove()
    # tg_model.load_state_dict(torch.load(ckp_name+'best'))
    return tg_model
