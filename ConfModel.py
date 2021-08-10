import torch.nn as nn
import torch as th
import torch.nn.functional as F
import pdb
from pytorch_revgrad import RevGrad

class ConfModelBC(nn.Module):

    def __init__(self, nOut, num_layers=2, **kwargs):
        super(ConfModelBC, self).__init__()
        self.out_features = nOut
        self.in_features = nOut*2
        self.num_layers = num_layers


        layers = [RevGrad()]

        layers.append(th.nn.Sequential(
            nn.BatchNorm1d(self.in_features),
            th.nn.ReLU(inplace=True),
            th.nn.Linear(self.in_features,512),
        ))

        layers.append(th.nn.Sequential(
            nn.BatchNorm1d(512),
            th.nn.ReLU(inplace=True),
            th.nn.Linear(512,2),
        ))

        self.matcher = th.nn.Sequential(*layers)

    def reset_parameters(self):
        self.matcher.reset_parameters()

    def forward(self, x):
        return self.matcher(x)