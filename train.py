import os
import sys
sys.path.append('../../Domain')
import itertools
import numpy as np
import torch
from custom_data import custom_dataset
from torch.utils.data import SubsetRandomSampler
from util import train
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
# dataset = MyCustomDataset(my_path)
batch_size = 1
validation_split = .2
shuffle_dataset = True
random_seed = 42






# 学习率全部缩十倍(最好的学习率是0.0001)
opt_lr = 0.0001
opt_b1 = 0.5
opt_b2 = 0.999
# Optimizers
# patch_net=ResNet_3d(Bottleneck, [3, 4, 6, 3],shortcut_type='B',no_cuda=False,num_classes=141,include_top=True).cuda()
# patch_net = ResNet2d(Bottleneck_2d, [3, 4, 6, 3], num_classes=1,include_top=True).cuda()
# patch_net=torch.load("patch_net.pth").cuda()

class CustomModel(nn.Module):
    def __init__(self, num_classes=65):
        super(CustomModel, self).__init__()
        self.resnet=models.resnet50(pretrained=False)
        self.resnet.load_state_dict(torch.load("resnet50-19c8e357.pth"), strict=True)
        modules = list(self.resnet.children())[:-1]  # 删除最后的全连接层
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Linear(num_ftrs+1, num_classes))
        self.resnet50 = nn.Sequential(*modules)
    def forward(self, x,initial_label):
        out_feat = self.resnet50(x).squeeze()

        mean = 0
        std_dev = 0.2
        indices = np.linspace(-1, 1, len(out_feat))
        gaussian_values = np.exp(-(indices - mean) ** 2 / (2 * std_dev ** 2))
        gaussian_values /= np.sum(gaussian_values)
        gaussian_tensor=torch.Tensor(gaussian_values[np.newaxis]).cuda()

        res=torch.mm(gaussian_tensor,out_feat)
        res=torch.cat((res,initial_label),1)
        out = self.resnet.fc(res)+initial_label

        return out

model = CustomModel(num_classes=1).cuda()
model.load_state_dict(torch.load("model.pth"))







loss_fn=torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, betas=(opt_b1, opt_b2))



dataset_feta = custom_dataset()
dataset_size = len(dataset_feta)
indices = list(range(dataset_size))
split_1 = int(np.floor(0.8 * dataset_size))
split_2 = int(np.floor(1.0 * dataset_size))
np.random.seed(42)
np.random.shuffle(indices)
train_indices, val_indices = indices[:split_1], indices[split_1:split_2]
valid_sampler = SubsetRandomSampler(train_indices)
train_data = torch.utils.data.DataLoader(dataset_feta, batch_size=1, sampler=valid_sampler)
valid_sampler = SubsetRandomSampler(val_indices)
val_data = torch.utils.data.DataLoader(dataset_feta, batch_size=1, sampler=valid_sampler)
cri_mse=torch.nn.MSELoss(reduction="mean")
cri_mse=torch.nn.L1Loss(reduction="mean")


print("***************************开始训练***************************")
train(model, train_data, val_data, 100, optimizer, cri_mse)
# 创建一个高斯分布的20*1的tensor,要求从中间到两侧的值按照高斯函数的形式递减