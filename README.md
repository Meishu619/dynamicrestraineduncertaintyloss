# Dynamic Restrained Uncertainty Loss

In this project, we are using raw audio from ICML ExVo challenge and melspectrograms as features. 

1. Architectures. We are using CNN as basic model. For each task, the CNN blocks are different. To reproduce this, you can use cnn14_dwa_un.py.
2. Losses. You can use cnn14_dwa_un.py and revised_dwa_uncertaintyloss.py together to achieve the benefits from revised_restrained_uncertaity_weighting_loss and dynamic_weigh_loss together.
