import torch.nn as nn
import torch
from torch.nn.functional import normalize

class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num, over_clustering=70):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num

        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )

        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j, x_w):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)
        h_w = self.resnet(x_w)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)
        z_w = normalize(self.instance_projector(h_w), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)
        c_w = self.cluster_projector(h_w)

        return z_i, z_j, z_w, c_i, c_j, c_w

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
