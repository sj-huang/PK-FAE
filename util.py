import tqdm
import torch
import SimpleITK as sitk
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

def cross_entropy_loss(predictions, targets):
    """
    :param predictions: 模型预测的概率值，shape为 (n_samples, n_classes)
    :param targets: 真实标签（one-hot形式），shape为 (n_samples, n_classes)
    :return: 交叉熵损失
    """
    assert predictions.shape == targets.shape, "Shapes of predictions and targets should be same"
    eps = 1e-15   # 添加一个极小值来避免log(0)的情况
    n_samples = predictions.shape[0]
    ce_loss = -torch.sum(targets * torch.log(predictions + eps)) / n_samples
    return ce_loss
import torch.nn.functional as F

def train(patch_net, train_loader, val_loader, N_EPOCHS, optimizer, loss_fn):
    min_loss=1e10

    with tqdm.tqdm(total=len(val_loader)) as pbar:
        with torch.no_grad():
            patch_net = patch_net.eval()
            metric=0
            for step, batch in enumerate(val_loader):
                image = batch[0][0].cuda().permute(0, 3, 1, 2)
                label = batch[1].cuda()
                initial_label = batch[2].cuda()
                paths_term = batch[3][0]

                pred_1 = patch_net(image,initial_label)

                loss = torch.nn.L1Loss(reduction='mean')(pred_1, label)
                metric+= loss

                pbar.update()
                pbar.set_description(f"Path_term {paths_term} | Loss: {loss.item()} | GT: {label.item()} | Pred: {pred_1.item()}")
            new_loss = metric / (step + 1)
            print("Min Loss: ", min_loss, "new Loss: ", new_loss)


import math
import csv
import torch
import tqdm
from scipy import stats

def test(patch_net, val_loader, csv_path="results.csv"):
    # 初始化 CSV 数据和表头
    results = []
    header = ["Path", "Prediction", "Label", "MSE Loss", "L1 Loss"]
    results.append(header)

    with tqdm.tqdm(total=len(val_loader)) as pbar:
        with torch.no_grad():
            patch_net = patch_net.eval()
            metric = 0
            mse_loss_fn = torch.nn.MSELoss(reduction='mean')  # 新增 MSE 损失函数
            l1_loss_fn = torch.nn.L1Loss(reduction='mean')

            for step, batch in enumerate(val_loader):
                image = batch[0][0].cuda().permute(0, 3, 1, 2)
                label = batch[1].cuda()
                initial_label = batch[2].cuda()
                paths_term = batch[3][0]

                pred_1 = patch_net(image, initial_label)

                # 计算 L1 和 MSE 损失
                l1_loss = l1_loss_fn(pred_1, label)
                mse_loss = mse_loss_fn(pred_1, label)
                metric += l1_loss

                # 将数据转换为可存储格式（例如 NumPy 或 Python 标量）
                pred_np = pred_1.cpu().numpy().flatten().tolist()  # 展平为列表
                label_np = label.cpu().numpy().flatten().tolist()
                
                # 添加到结果列表
                row = [
                    paths_term,
                    pred_np,
                    label_np,
                    mse_loss.item(),
                    l1_loss.item()
                ]
                results.append(row)

                pbar.update()
                pbar.set_description(f"Path_term {paths_term} Loss: {l1_loss.item()}")

            new_loss = metric / (step + 1)
            print("Final Average L1 Loss: ", new_loss.item())

    # 写入 CSV 文件
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)
    
    return new_loss