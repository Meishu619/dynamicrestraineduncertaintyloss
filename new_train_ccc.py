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

# from models import (
#     Cnn10,
#     Cnn14
# )
from age_shallow1 import (
    Cnn10,
    Cnn14
)

BATCH_SIZE = 3
LR = 0.001
EPOCH = 1000

home = "/home/meishu/eihw/data_work/"
nas = "/nas/staff/data_work/"

path = home

file_path = path + "Meishu/0_ExVo22/baseline/feats/mel_2.5/"
train_path = path + "/Meishu/0_ExVo22/exvo_train.csv"
val_path = path + "/Meishu/0_ExVo22/exvo_val.csv"

torch.autograd.set_detect_anomaly(True)

train_set = dataloader_new.AudioSet(file_path, train_path)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)


val_set = dataloader_new.AudioSet(file_path, val_path)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
class CCCLoss(torch.nn.Module):
    def forward(self, output, target):
        out_mean = torch.mean(output)
        target_mean = torch.mean(target)

        covariance = torch.mean((output - out_mean) * (target - target_mean))
        target_var = torch.mean((target - target_mean)**2)
        out_var = torch.mean((output - out_mean)**2)

        ccc = 2.0 * covariance / \
            (target_var + out_var + (target_mean - out_mean)**2 + 1e-10)
        loss_ccc = 1.0 - ccc

        return loss_ccc


# a: age, e: emotions, c: country
loss_a = nn.L1Loss()
loss_e = CCCLoss()
loss_c = nn.CrossEntropyLoss()

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Cnn14(output_dim=12, n=[1,6,1]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)  # optimize all parameters

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
        data = data.to(device)

        # data = data.unsqueeze(dim=1)

        Country = Country.to(device).long()
        Age = Age.unsqueeze(-1).to(device)
        Emotions = Emotions.float().clone().detach().requires_grad_(True).to(device)
        # Emotions = torch.tensor(Emotions).to(device).float()

        # Define different outputs # Should be done
        [emotion, country, age], logsigma = model(data)
        loss_age = loss_a(age, Age)
        loss_country = loss_c(country, Country)
        # Compute Country UAR
        country_truth.append(Country.detach().cpu().numpy())
        country_pred.append(torch.argmax(country, dim=1).detach().cpu().numpy())

        loss_emotions = loss_e(emotion, Emotions)

        loss = loss_age * logsigma[0] + loss_country * logsigma[1] + loss_emotions * logsigma[2]

        optimizer.zero_grad()
        loss.backward()
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

        if idx == 4:
            break

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
            [emotion, country, age], logsigma = model(data)
            loss_age = loss_a(age, Age)
            loss_country = loss_c(country, Country)
            # Compute Country UAR
            country_truth_v.append(Country.detach().cpu().numpy())
            country_pred_v.append(torch.argmax(country, dim=1).detach().cpu().numpy())

            loss_emotions = loss_e(emotion, Emotions)
            if idx == 4:
                break

            loss = loss_age * logsigma[0] + loss_country * logsigma[1] + loss_emotions * logsigma[2]

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













