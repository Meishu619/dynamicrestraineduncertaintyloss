import torch
import torch.nn.functional as F


weight_age = F.softmax(torch.randn(6), dim=-1)
values_age = torch.zeros_like(weight_age)
weight_age = torch.argmax(weight_age)
values_age[weight_age] = 1

print(values_age)