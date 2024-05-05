# -*- coding: utf-8 -*-
# @Time    : 2024/4/29 下午11:00
# @Author  : yang chen
import os
from Utils.ReadDataset import FUSAR_DATASET,FUSAR_DATASET_CONFIG,getFileList
from torch.utils.data import DataLoader

if __name__ == '__main__':
    PROJECT_HOME = os.getcwd()
    dataset_root = os.path.join(PROJECT_HOME, 'Dataset', 'FUSAR')
    fusar_config = FUSAR_DATASET_CONFIG(dataset_root)
    img_list = getFileList(os.path.join(fusar_config.IMAGE_ROOT, '*.tif'))
    mask_list = getFileList(os.path.join(fusar_config.MASK_ROOT, '*.tif'))
    classes = fusar_config.CLASSES
    colormap = fusar_config.COLORMAP
    fusar_dataset = FUSAR_DATASET(img_list, mask_list, classes, colormap)
    train_loader = DataLoader(fusar_dataset,batch_size=2, shuffle=True)
    for epoch in range(10):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            print(batch_idx)
            print(inputs.shape)
            print(targets.shape)
            print(123)
    print(123)

