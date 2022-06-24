import torch
import torch.nn as nn
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

class UncertaintyRevised(nn.Module):

    def __init__(self):
        # super(Uncertainty, self).__init__()
        super().__init__()

        # self.log_vars = nn.Parameter(torch.FloatTensor([a, b, c]))

        self.log_vars = nn.Parameter(torch.FloatTensor([0.33, 0.33, 0.33]))

    def forward(self, preds, emo, cou, age):
        emo_mse, cou_cross, age_mae = CCCLoss(), nn.CrossEntropyLoss(), nn.L1Loss()
        loss_emo = emo_mse(preds[0], emo)
        loss_cou = cou_cross(preds[1], cou)
        loss_age = age_mae(preds[2], age)

        loss_emo = 0.33 / (self.log_vars[0] ** 2) * loss_emo + torch.log(1 + self.log_vars[0] ** 2)
        loss_cou = 0.33 / (self.log_vars[1] ** 2) * loss_cou + torch.log(1 + self.log_vars[1] ** 2)
        loss_age = 0.33 / (self.log_vars[2] ** 2) * loss_age + torch.log(1 + self.log_vars[2] ** 2)

        weights = [0.33 / (self.log_vars[0] ** 2), 0.33 / (self.log_vars[1] ** 2), 0.33 / (self.log_vars[2] ** 2)]
        loss_weight = torch.abs(
            1 - torch.abs(self.log_vars[0]) - torch.abs(self.log_vars[1]) - torch.abs(self.log_vars[2]))
        return loss_emo + loss_cou + loss_age+ loss_weight, weights