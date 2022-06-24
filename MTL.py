import torch, argparse
import torch.nn as nn
import torch.nn.functional as F
from age_shallow1 import (
    Cnn10,
    Cnn14
)
# from LibMTL import Trainer
# from LibMTL.model import resnet_dilated
# from LibMTL.utils import set_random_seed, set_device
# from LibMTL.config import LibMTL_args, prepare_args
# import LibMTL.weighting as weighting_method
# import LibMTL.architecture as architecture_method
from LibMTL.weighting import GradNorm

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import os
from sklearn.metrics import recall_score
import numpy as np
import torch
import torch.nn as nn
import dataloader_new
from torch.utils.data import DataLoader
from utils import EvalMetrics
from scipy.stats import hmean
import torch.nn.functional as F
from revised_uncertaintyloss import UncertaintyRevised
from LibMTL.weighting.abstract_weighting import AbsWeighting


class GLS(AbsWeighting):
    r"""Geometric Loss Strategy (GLS).

    This method is proposed in `MultiNet++: Multi-Stream Feature Aggregation and Geometric Loss Strategy for Multi-Task Learning (CVPR 2019 workshop) <https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf>`_ \
    and implemented by us.
    """

    def __init__(self):
        super(GLS, self).__init__()

    def backward(self, losses, **kwargs):
        loss = torch.pow(losses.prod(), 1. / self.task_num)
        loss.backward()
        batch_weight = losses / (self.task_num * losses.prod())
        return batch_weight.detach().cpu().numpy()
class RGradNorm(GradNorm):
    def __init__(self):
        super(RGradNorm, self).__init__()
        self.epoch = 0
        self.task_num = 3
        self.device = device
        self.init_param()

    def backward(self, losses, **kwargs):
        self.epoch = kwargs['epoch']
        self.task_num = len(losses)
        self.rep_grad = False
        self.train_loss_buffer = kwargs['train_loss_buffer']
        self.device = kwargs['device']
        return super(RGradNorm, self).backward(losses, alpha=kwargs['alpha'])


class IMTL(AbsWeighting):
    r"""Impartial Multi-task Learning (IMTL).

    This method is proposed in `Towards Impartial Multi-task Learning (ICLR 2021) <https://openreview.net/forum?id=IMPnRXEWpvr>`_ \
    and implemented by us.
    """

    def __init__(self):
        super(IMTL, self).__init__()

    def init_param(self):
        self.loss_scale = nn.Parameter(torch.tensor([0.0] * self.task_num, device=self.device))

    def backward(self, losses, **kwargs):
        losses = self.loss_scale.exp() * losses - self.loss_scale
        grads = self._get_grads(losses, mode='backward')
        if self.rep_grad:
            per_grads, grads = grads[0], grads[1]

        grads_unit = grads / torch.norm(grads, p=2, dim=-1, keepdim=True)

        D = grads[0:1].repeat(self.task_num - 1, 1) - grads[1:]
        U = grads_unit[0:1].repeat(self.task_num - 1, 1) - grads_unit[1:]

        alpha = torch.matmul(torch.matmul(grads[0], U.t()), torch.inverse(torch.matmul(D, U.t())))
        alpha = torch.cat((1 - alpha.sum().unsqueeze(0), alpha), dim=0)

        if self.rep_grad:
            self._backward_new_grads(alpha, per_grads=per_grads)
        else:
            self._backward_new_grads(alpha, grads=grads)
        return alpha.detach().cpu().numpy()


class GLS(AbsWeighting):
    r"""Geometric Loss Strategy (GLS).

    This method is proposed in `MultiNet++: Multi-Stream Feature Aggregation and Geometric Loss Strategy for Multi-Task Learning (CVPR 2019 workshop) <https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf>`_ \
    and implemented by us.
    """

    def __init__(self):
        super(GLS, self).__init__()

    def backward(self, losses, **kwargs):
        loss = torch.pow(losses.prod(), 1. / self.task_num)
        loss.backward()
        batch_weight = losses / (self.task_num * losses.prod())
        return batch_weight.detach().cpu().numpy()

# from models import (
#     Cnn10,
#     Cnn14
# )
from age_shallow1 import (
    Cnn10,
    Cnn14
)

# log = [0.1, 0.2, 0.7]
# print("weights ratial")
# print(log)

print("Uncertainty without line")

number = [1, 3, 5]
print("block num")
print(number)

BATCH_SIZE = 32
LR = 0.001
EPOCH = 200

home = "/home/meishu/eihw/data_work/"
nas = "/nas/staff/data_work/"

path = home

# file_path = "/data/eihw-gpu6/songmeis/mel_2.5/"
# train_path = path + "/Meishu/0_ExVo22/exvo_train.csv"
# val_path = path + "/Meishu/0_ExVo22/exvo_val.csv"

file_path = path + "Meishu/0_ExVo22/baseline/feats/mel_2.5/"
train_path = path + "/Meishu/0_ExVo22/exvo_train.csv"
val_path = path + "/Meishu/0_ExVo22/exvo_val.csv"


torch.autograd.set_detect_anomaly(True)

train_set = dataloader_new.AudioSet(file_path, train_path)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                           drop_last=True)

val_set = dataloader_new.AudioSet(file_path, val_path)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

# a: age, e: emotions, c: country
loss_a = nn.L1Loss()
loss_e = nn.MSELoss()
loss_c = nn.CrossEntropyLoss()

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Cnn14(output_dim=12, n=number).to(device)

# loss_function = Uncertainty().to(device)
# loss_function = GLS().to(device)
mtl_loss_fn = GLS()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)  # optimize all parameters

print(f"model {model} | optimizer {optimizer} ")

train_loss_buffer = np.zeros([3, EPOCH])
for epoch in range(EPOCH):
    print(f"EPOCHS {epoch + 1} | LR {LR} | Batch Size {BATCH_SIZE}")

    tmp_losses = 0.
    tmp_counter = 0
    losses = 0.
    age_losses = 0.
    emotion_losses = 0.
    country_losses = 0.
    counter = 1

    country_pred = []
    country_truth = []

    emotion_label = []

    ccc_result = []

    age_mae = 0.0

    # add ccc new calculation
    # Emotion labels
    Emotions_all = []
    # emotion data
    emotion_all = []

    model.train()
    for idx, (data, Subject_ID, Country, Country_string, Age, Emotions) in enumerate(train_loader):
        data = data.to(device)
        if idx == 4:
            break

        # data = data.unsqueeze(dim=1)

        Country = Country.to(device).long()
        Age = Age.unsqueeze(-1).to(device)
        Emotions = Emotions.float().clone().detach().requires_grad_(True).to(device)
        # Emotions = torch.tensor(Emotions).to(device).float()

        # Define different outputs # Should be done
        preds, logsigma = model(data)
        [emotion, country, age] = preds
        loss_age = loss_a(age, Age)
        loss_country = loss_c(country, Country)
        # Compute Country UAR
        country_truth.append(Country.detach().cpu().numpy())
        country_pred.append(torch.argmax(country, dim=1).detach().cpu().numpy())

        loss_emotions = loss_e(emotion, Emotions)

        loss_list = torch.stack([loss_age, loss_country, loss_emotions])
        train_loss_buffer[:, epoch] = loss_list.detach().numpy()


        # loss = loss_function(preds, Emotions, Country, Age)
        # weight = F.softmax(torch.randn(3), dim=-1).to(device)  # RLW is only this!
        #
        # loss = torch.sum(loss * weight)
        # loss = loss_age  + loss_country  + loss_emotions

        optimizer.zero_grad()
        loss_weight_array = mtl_loss_fn.backward(loss_list, alpha=1.5, epoch=epoch,
                                                 train_loss_buffer=train_loss_buffer, device=device)

        # loss.backward()

        optimizer.step()

        # losses += (loss_list * loss_weight_array).sum().item()
        # # losses += loss.item()
        # age_losses += (loss_list[0] * loss_weight_array[0]).item()
        # age_mae = loss_age
        # emotion_losses += (loss_list[1] * loss_weight_array[1]).item()
        # country_losses += (loss_list[2] * loss_weight_array[2]).item()
        print(loss_list)
        print(loss_weight_array)
        counter += 1

        # save emotion data
        Emotions_all.append(Emotions.detach().cpu().numpy())
        emotion_all.append(emotion.detach().cpu().numpy())

    # classes = ["a", "b", "v", "d", "e", "f", "g", "h", "i", "j"]
    # TODO: Considering the situation when batch size is not 1
    E = np.stack(Emotions_all)
    e = np.stack(emotion_all)
    for identifier in range(10):
        # identifier = classes.index(j)
        ccc = EvalMetrics.CCC(
            E[:, :, identifier].flatten(),
            e[:, :, identifier].flatten()
        )
        ccc_result.append(ccc)

    country_pred = np.concatenate(country_pred, axis=0)
    country_truth = np.concatenate(country_truth, axis=0)

    UAR = recall_score(country_truth, country_pred, average="macro")

    # val_hmean_score = hmean([np.mean(ccc_result.cpu().detach().numpy()), UAR, 1/age_mae])
    #
    #
    # print(f"HMean: {np.round(val_hmean_score, 4)}\n------")
    print(
        f"age loss: {age_losses / counter}\t country loss: {country_losses / counter} \t emotion loss: {emotion_losses / counter}")
    print(f"Loss Weights: age--{logsigma[0]}\t country--{logsigma[1]}\t emotions--{logsigma[2]}")
    print(
        f"{epoch + 1}/{EPOCH}\t EmoCCC: {np.round(np.mean(ccc_result), 3)}\tCountryUAR: {np.round(UAR, 3)}\t AgeMAE: {np.round(age_losses / counter, 3)}\tTrainLoss: {losses / counter}")

    # model.eval()
    # tmp_losses = 0.
    # tmp_counter = 0
    # losses = 0.
    # age_losses = 0.
    # emotion_losses = 0.
    # country_losses = 0.
    # counter = 1
    #
    # emotion_label = []
    #
    # ccc_result = []
    #
    # age_mae = 0.0
    #
    # country_truth_v = []
    # country_pred_v = []
    #
    # # add ccc new calculation
    # # Emotion labels
    # Emotions_all_v = []
    # # emotion data
    # emotion_all_v = []
    #
    # with torch.no_grad():
    #     for idx, (data, Subject_ID, Country, Country_string, Age, Emotions) in enumerate(val_loader):
    #         data = data.to(device)
    #         # data = data.unsqueeze(dim=1)
    #
    #         Country = Country.to(device).long()
    #         Age = Age.to(device)
    #
    #         Emotions = torch.tensor(Emotions).to(device).float()
    #
    #         # Define different outputs # Should be done
    #         preds, logsigma = model(data)
    #         [emotion, country, age] = preds
    #         loss_age = loss_a(age, Age)
    #         loss_country = loss_c(country, Country)
    #         # Compute Country UAR
    #         country_truth_v.append(Country.detach().cpu().numpy())
    #         country_pred_v.append(torch.argmax(country, dim=1).detach().cpu().numpy())
    #
    #         loss_emotions = loss_e(emotion, Emotions)
    #
    #         # loss = loss_age + loss_country + loss_emotions
    #         # loss = loss_function(preds, Emotions, Country, Age)
    #         weight = F.softmax(torch.randn(3), dim=-1).to(device)  # RLW is only this!
    #
    #         loss = torch.sum(loss * weight)
    #         losses += loss.item()
    #         age_losses += loss_age.item()
    #         age_mae = loss_age
    #         emotion_losses += loss_emotions.item()
    #         country_losses += loss_country.item()
    #         counter += 1
    #         # save emotion data
    #         Emotions_all_v.append(Emotions.detach().cpu().numpy())
    #         emotion_all_v.append(emotion.detach().cpu().numpy())
    #
    #     classes = ["a", "b", "v", "d", "e", "f", "g", "h", "i", "j"]
    #     E_v = np.stack(Emotions_all_v)
    #     e_v = np.stack(emotion_all_v)
    #     for j in classes:
    #         identifier = classes.index(j)
    #         ccc = EvalMetrics.CCC(
    #             E_v[:, :, identifier].flatten(),
    #             e_v[:, :, identifier].flatten()
    #         )
    #         ccc_result.append(ccc)
    #
    #     country_pred_v = np.concatenate(country_pred_v, axis=0)
    #     country_truth_v = np.concatenate(country_truth_v, axis=0)
    #     UAR = recall_score(country_truth_v, country_pred_v, average="macro")
    #     print(
    #         f"age loss: {age_losses / counter}\t country loss: {country_losses / counter} \t emotion loss: {emotion_losses / counter}")
    #     print(
    #         f"{epoch + 1}/{EPOCH}\t ValEmoCCC: {np.round(np.mean(ccc_result), 3)}\tValCountryUAR: {np.round(UAR, 3)}\t ValAgeMAE: {np.round(age_losses / counter, 3)}\tValLoss: {losses / counter}")
    #     print(f"H-Mean Score:{hmean([np.mean(ccc_result), UAR, 1 / (age_losses / counter)])}")












