# -*- coding: utf-8 -*-
# @Time    : 2024/4/29 下午9:49
# @Author  : yang chen
import os.path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
from Utils.AugFunction import to_tensor
import cv2
class FUSAR_DATASET_CONFIG:
    def __init__(self,dataset_root):
        self.IMAGE_ROOT= os.path.join(dataset_root,'SAR_1024')
        self.MASK_ROOT= os.path.join(dataset_root,'LAB_1024')
        self.CLASSES_NAME = ['Land', 'Building', 'Vegetation', 'Water', 'Road']
        self.CLASSES_COLORMAP = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # black,red,green,blue,yellow
        self.CLASSES_INDEX = [0,1,2,3,4]    # 五类
        self.CLASSES_INDEX = [0, 1, 0, 2, 3]  # 四类
        self.NUM_CLASSES = len(np.unique(self.CLASSES_INDEX))  # COLORMAP是一个二维矩阵，C_NUM,C_COLOR 因此axis=0

def getFileList(path, is_sort=True):
    fileList = glob(path)
    if is_sort:
        fileList = sorted(fileList)
    return fileList

def process_mask(rgb_mask, classes,classes_idx,colormap):
    '''
    将彩色mask映射到对应的类别索引
    期待的rgb_mask尺寸 为(h,w,3)
         classes     为类名
         classes_idx 为可重复的类别索引
         colormap    为三通道的类别值
    '''
    output_mask = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)

    for class_idx, color in enumerate(colormap):
        output_mask[np.all(np.equal(rgb_mask, color), axis=-1)] = classes_idx[class_idx]

    # # debug - 该循环为了演示每一类标签图
    # for idx,class_idx in enumerate(classes_idx):
    #     plt.title('{}'.format(classes[idx]))
    #     plt.imshow(np.equal(output_mask, class_idx), cmap='gray')
    #     plt.show()
    # # debug - 该循环为了演示每一类标签图
    return output_mask

class FUSAR_DATASET(Dataset):
    def __init__(self, img_list, mask_list, classes, classes_idx ,colormap, resize = 256):
        self.img_list = img_list
        self.mask_list = mask_list
        self.classes = classes
        self.classes_idx = classes_idx
        self.colormap = colormap
        self.resize = resize

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img =  cv2.imread(self.img_list[idx],cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(self.resize,self.resize))
        mask = cv2.imread(self.mask_list[idx],cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask,(self.resize,self.resize))
        ## debug - show img and mask
        # plt.imshow(img,cmap='gray')
        # plt.title('Original Image')
        # plt.show()
        # plt.imshow(mask)
        # plt.title('Original Label')
        # plt.show()
        ## debug - show img and mask

        processed_mask = process_mask(mask,self.classes,self.classes_idx,self.colormap)
        # # debug - show processed mask
        # plt.imshow(processed_mask, cmap='gray')
        # plt.title('Processed Label')
        # plt.show()
        # # debug - show processed mask

        # img and mask are augmented by a set of functions
        # augment function

        # img/mask should be converted to Tensor before inputting it into Model
        img = to_tensor(img)
        processed_mask = to_tensor(processed_mask,is_mask=True)
        return img,processed_mask

# # debug
# project_home = r'D:\RS_WorkSpace\SARImageSegmentation_HOME'
# dataset_root = os.path.join(project_home,'Dataset','FUSAR')
# fusar_config = FUSAR_DATASET_CONFIG(dataset_root)
# img_list = getFileList(os.path.join(fusar_config.IMAGE_ROOT,'*.tif'))
# mask_list = getFileList(os.path.join(fusar_config.MASK_ROOT,'*.tif'))
# classes = fusar_config.CLASSES_NAME
# classes_idx = fusar_config.CLASSES_INDEX
# colormap = fusar_config.CLASSES_COLORMAP
# fusar_dataset = FUSAR_DATASET(img_list,mask_list,classes,classes_idx,colormap)
# img , mask = fusar_dataset[102]
# print(img.shape)
# print(mask.shape)


