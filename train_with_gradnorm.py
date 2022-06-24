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
from loss_func_ori import Ori_loss_func

# from models import (
#     Cnn10,
#     Cnn14
# )
from cnn14_an import (
    Cnn14_train,
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
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)


val_set = dataloader_new.AudioSet(file_path, val_path)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

# a: age, e: emotions, c: country
loss_a = nn.L1Loss()
loss_e = nn.MSELoss()
loss_c = nn.CrossEntropyLoss()

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Cnn14_train(Cnn14())



# loss_function = Uncertainty().to(device)
loss_function = Ori_loss_func().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)  # optimize all parameters

print(f"model {model} | optimizer {optimizer} ")


for epoch in range(EPOCH):
    print(f"EPOCHS {epoch+1} | LR {LR} | Batch Size {BATCH_SIZE}")

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
        if idx == 4:
            break
        data = data.to(device)

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

        loss, weights = loss_function(preds, Emotions, Country, Age)
        # weight = F.softmax(torch.randn(3), dim=-1).to(device)  # RLW is only this!
        #
        # loss = torch.sum(loss * weight)
        # loss = loss_age  + loss_country  + loss_emotions

        optimizer.zero_grad()
        loss.backward()

        #####################
        task_loss = torch.stack([loss_age, loss_country, loss_emotions]).to(device)
        if epoch == 0:
            # set L(0)
            if torch.cuda.is_available():
                initial_task_loss = task_loss.data.cpu()
            else:
                initial_task_loss = task_loss.data
            initial_task_loss = initial_task_loss.numpy()

        weights.grad.data = weights.grad.data * 0.0
        if True:
            # get layer of shared weights


            # get the gradient norms for each of the tasks
            # G^{(i)}_w(t)
            norms = []
            for i in range(len(task_loss)):
                # get the gradient of this task loss with respect to the shared parameters
                gygw = torch.autograd.grad(task_loss[i], weights, retain_graph=True)
                # compute the norm
                norms.append(torch.norm(torch.mul(weights[i], gygw[0])))
            norms = torch.stack(norms)
            # print('G_w(t): {}'.format(norms))

            # compute the inverse training rate r_i(t)
            # \curl{L}_i
            if torch.cuda.is_available():
                loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
            else:
                loss_ratio = task_loss.data.numpy() / initial_task_loss
            # r_i(t)
            inverse_train_rate = loss_ratio / np.mean(loss_ratio)
            # print('r_i(t): {}'.format(inverse_train_rate))

            # compute the mean norm \tilde{G}_w(t)
            if torch.cuda.is_available():
                mean_norm = np.mean(norms.data.cpu().numpy())
            else:
                mean_norm = np.mean(norms.data.numpy())
            # print('tilde G_w(t): {}'.format(mean_norm))

            # compute the GradNorm loss
            # this term has to remain constant
            constant_term = torch.tensor(mean_norm * (inverse_train_rate ** 0.12), requires_grad=False)
            if torch.cuda.is_available():
                constant_term = constant_term.cuda()
            # print('Constant term: {}'.format(constant_term))
            # this is the GradNorm loss itself
            grad_norm_loss = torch.tensor(torch.sum(torch.abs(norms - constant_term)))
            # print('GradNorm loss {}'.format(grad_norm_loss))

            # compute the gradient for the weights
            weights.grad = torch.autograd.grad(grad_norm_loss, weights)[0]
        #####################

        optimizer.step()

        losses += loss.item()
        age_losses += loss_age.item()
        age_mae = loss_age
        emotion_losses += loss_emotions.item()
        country_losses += loss_country.item()
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
    print(f"{epoch + 1}/{EPOCH}\t EmoCCC: {np.round(np.mean(ccc_result), 3)}\tCountryUAR: {np.round(UAR, 3)}\t AgeMAE: {np.round(age_losses / counter, 3)}\tTrainLoss: {losses / counter}")

    model.eval()
    tmp_losses = 0.
    tmp_counter = 0
    losses = 0.
    age_losses = 0.
    emotion_losses = 0.
    country_losses = 0.
    counter = 1

    emotion_label = []

    ccc_result = []

    age_mae = 0.0

    country_truth_v = []
    country_pred_v = []

    # add ccc new calculation
    # Emotion labels
    Emotions_all_v = []
    # emotion data
    emotion_all_v = []

    with torch.no_grad():
        for idx, (data, Subject_ID, Country, Country_string, Age, Emotions) in enumerate(val_loader):
            data = data.to(device)
            # data = data.unsqueeze(dim=1)

            Country = Country.to(device).long()
            Age = Age.to(device)

            Emotions = torch.tensor(Emotions).to(device).float()

            # Define different outputs # Should be done
            preds, logsigma = model(data)
            [emotion, country, age] = preds
            loss_age = loss_a(age, Age)
            loss_country = loss_c(country, Country)
            # Compute Country UAR
            country_truth_v.append(Country.detach().cpu().numpy())
            country_pred_v.append(torch.argmax(country, dim=1).detach().cpu().numpy())

            loss_emotions = loss_e(emotion, Emotions)


            # loss = loss_age + loss_country + loss_emotions
            loss = loss_function(preds, Emotions, Country, Age)
            weight = F.softmax(torch.randn(3), dim=-1).to(device)  # RLW is only this!

            loss = torch.sum(loss * weight)
            losses += loss.item()
            age_losses += loss_age.item()
            age_mae = loss_age
            emotion_losses += loss_emotions.item()
            country_losses += loss_country.item()
            counter += 1
            # save emotion data
            Emotions_all_v.append(Emotions.detach().cpu().numpy())
            emotion_all_v.append(emotion.detach().cpu().numpy())

        classes = ["a", "b", "v", "d", "e", "f", "g", "h", "i", "j"]
        E_v = np.stack(Emotions_all_v)
        e_v = np.stack(emotion_all_v)
        for j in classes:
            identifier = classes.index(j)
            ccc = EvalMetrics.CCC(
                E_v[:, :, identifier].flatten(),
                e_v[:, :, identifier].flatten()
            )
            ccc_result.append(ccc)

        country_pred_v = np.concatenate(country_pred_v, axis=0)
        country_truth_v = np.concatenate(country_truth_v, axis=0)
        UAR = recall_score(country_truth_v, country_pred_v, average="macro")
        print(
            f"age loss: {age_losses / counter}\t country loss: {country_losses / counter} \t emotion loss: {emotion_losses / counter}")
        print(
            f"{epoch + 1}/{EPOCH}\t ValEmoCCC: {np.round(np.mean(ccc_result), 3)}\tValCountryUAR: {np.round(UAR, 3)}\t ValAgeMAE: {np.round(age_losses / counter, 3)}\tValLoss: {losses / counter}")
        print(f"H-Mean Score:{hmean([np.mean(ccc_result), UAR, 1/(age_losses / counter)])}")












