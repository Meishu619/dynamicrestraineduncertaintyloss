import torch, argparse
import torch.nn as nn
import torch.nn.functional as F

from LibMTL._record import _PerformanceMeter
from utils import *
from cnn14_an import (
    Cnn10,
    Cnn14
)
import dataloader_new
from LibMTL import Trainer
from LibMTL.model import resnet_dilated
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import os
from sklearn.metrics import recall_score
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import EvalMetrics
from scipy.stats import hmean

home = "/home/meishu/eihw/data_work/"
nas = "/nas/staff/data_work/"

path = home

# file_path = "/data/eihw-gpu6/songmeis/mel_2.5/"
# train_path = path + "/Meishu/0_ExVo22/exvo_train.csv"
# val_path = path + "/Meishu/0_ExVo22/exvo_val.csv"

file_path = path + "Meishu/0_ExVo22/baseline/feats/mel_2.5/"
train_path = path + "/Meishu/0_ExVo22/exvo_train.csv"
val_path = path + "/Meishu/0_ExVo22/exvo_val.csv"

def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--train_mode', default='trainval', type=str, help='trainval, train')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    return parser.parse_args()

number = [1, 3, 5]
print("block num")
print(number)

BATCH_SIZE = 1
LR = 0.001
EPOCH = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    train_set = dataloader_new.AudioSet(file_path, train_path)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                               drop_last=True)
    print(len(train_loader))

    val_set = dataloader_new.AudioSet(file_path, val_path)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                                             drop_last=True)
    print(len(val_loader))

    # define tasks
    task_dict = {'age': {'metrics': ['mIoU', 'pixAcc'],
                                  'metrics_fn': nn.L1Loss(),
                                  'loss_fn': nn.L1Loss(),
                                  'weight': [0, 0]},
                 'country': {'metrics': ['abs_err', 'rel_err'],
                           'metrics_fn': nn.CrossEntropyLoss(),
                           'loss_fn': nn.CrossEntropyLoss(),
                           'weight': [0, 0]},
                 'emotion': {'metrics': ['mean', 'median', '<11.25', '<22.5', '<30'],
                            'metrics_fn': nn.MSELoss(),
                            'loss_fn': nn.MSELoss(),
                            'weight': [0, 0, 0, 0, 0]}}

    # define encoder and decoders
    def encoder_class():
        return resnet_dilated('resnet50')

    num_out_channels = {'age': 1, 'country': 4, 'emotion': 10}
    model = Cnn14(output_dim=12)
    decoders = {"age": model}
    class NYUtrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class, decoders,
                     rep_grad, multi_input, optim_param, scheduler_param, **kwargs):
            super(Trainer, self).__init__()

            self.device = device
            self.kwargs = kwargs
            self.task_dict = task_dict
            self.task_num = len(task_dict)
            self.task_name = list(task_dict.keys())
            self.rep_grad = rep_grad
            self.multi_input = multi_input

            self._prepare_model(weighting, architecture, encoder_class, decoders)
            self._prepare_optimizer(optim_param, scheduler_param)

            self.meter = _PerformanceMeter(self.task_dict, self.multi_input)
        #
        # def process_preds(self, preds):
        #     img_size = (288, 384)
        #     for task in self.task_name:
        #         preds[task] = F.interpolate(preds[task], img_size, mode='bilinear', align_corners=True)
        #     return preds

    NYUmodel = NYUtrainer(task_dict=task_dict,
                          weighting=params.weighting,
                          architecture=params.arch,
                          encoder_class=nn.Identity,
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param)
    NYUmodel.train(train_loader, val_loader, 200)


if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    # set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)