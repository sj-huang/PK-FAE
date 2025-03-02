from torch.utils.data import Dataset
import pickle
import os
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
import SimpleITK as sitk
import pandas as pd
from PIL import Image
import imageio
"""
    定义一个 读取数据集.pkl文件的类：
"""
def get_patch(image):

    patch_size = 32  # patch 的大小
    stride = 32  # 步幅

    # 划分图像为 patch
    patches = []
    for s in range(len(image)):
        for i in range(0, 256 - patch_size + 1, stride):
            for j in range(0, 256 - patch_size + 1, stride):
                patch = image[s,i:i + patch_size, j:j + patch_size]
                patches.append(patch)

    return np.array(patches)
# 定义一个子类叫 custom_dataset，继承与 Dataset
class custom_dataset(Dataset):
    def __init__(self):
        self.image=[]
        self.label=[]
        self.initial_label=[]
        self.paths_name=[]

        csv_path = "Age_day_all.xlsx"
        csv_file = pd.read_excel(csv_path)
        patient_id = list(csv_file.iloc[:, 0])
        csv_label = csv_file.iloc[:, 1]
        csv_initial_label = csv_file.iloc[:, 2]

        root_paths = r"/public_bme/data/huangshj/Age/Ours/LR_128_4"
        paths_list = os.listdir(root_paths)
        for paths in paths_list:
            root_path = os.path.join(root_paths, paths)
            path_lists = os.listdir(root_path)

            for pat in path_lists:
                if paths=="zhejiang_1":
                    term = "MR" + pat.split("_")[0]
                else:
                    term = pat.split("_")[0]

                index = patient_id.index(term)
                label_id = csv_label[index]
                initial_label_id = csv_initial_label[index]



                # gw=int(pat.split("_")[2])
                pat = os.path.join(root_path,pat)
                pat_lists = os.listdir(pat)

                for pa in pat_lists:

                    axi_view = []
                    pa = os.path.join(pat, pa)
                    axi_img = sitk.ReadImage(pa)
                    axi_img=sitk.GetArrayFromImage(axi_img)
                    for i in range(axi_img.shape[0]):
                        slice=axi_img[i]
                        imageio.imwrite("slice.png", slice)
                        slice = cv2.imread("slice.png")
                        axi_view.append(slice)

                    self.image.append(axi_view)
                    label = np.zeros([1])
                    label[0] = label_id
                    self.label.append(label)

                    paths_term = pa.split(".nii")[0]

                    initial_label = np.zeros([1])
                    initial_label[0] = initial_label_id
                    self.initial_label.append(initial_label)
                    self.paths_name.append(paths_term)





    def __getitem__(self, idx):  # 根据 idx 取出其中一个name
        image = self.image[idx % len(self.image)]
        label = self.label[idx % len(self.label)]
        paths_term = self.paths_name[idx % len(self.paths_name)]
        initial_label = self.initial_label[idx % len(self.initial_label)]
        degree = np.random.randint(0, 360, dtype='int')
        img_list=np.zeros((len(image),192,192,3))
        for i,slice in enumerate(image):
            slice=Image.fromarray(slice)
            slice = slice.rotate(degree)
            # slice=slice.resize((192,192))
            slice = np.array(slice)
            slice = (slice - slice.min()) / (slice.max() - slice.min() + 1e-7)
            img_list[i]=slice
        image=torch.Tensor(img_list).cuda()


        return image,torch.Tensor(label).cuda(),torch.Tensor(initial_label).cuda(),paths_term
    def __len__(self):  # 总数据的多少
        return len(self.label)


def read_train(path):
    train_dataset = custom_dataset(path)
    train_data = DataLoader(train_dataset, 1, shuffle=True)
    return train_data

