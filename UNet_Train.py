# -*- coding: utf-8 -*-
# @Time    : 2024/4/29 下午11:00
# @Author  : yang chen
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Utils.ReadDataset import FUSAR_DATASET,FUSAR_DATASET_CONFIG,getFileList
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Models.UNet.UNet import UNet
from Utils.Tools import getTime
from Utils.Logger import Logger
from Utils.SegMetrics import SegMetrics

if __name__ == '__main__':

    # ROOT PATH
    PROJECT_HOME = os.getcwd()

    # DATAPATH ROOT PATH
    dataset_root = os.path.join(PROJECT_HOME, 'Dataset', 'FUSAR')

    # OUTPUT ROOT PATH
    prefix = str(getTime())
    output_root = os.path.join(PROJECT_HOME,'Output', prefix)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    logger = Logger(log_dir=output_root)

    # READ IMAGE/MASK PATH LIST
    fusar_config = FUSAR_DATASET_CONFIG(dataset_root)
    img_list = getFileList(os.path.join(fusar_config.IMAGE_ROOT, '*.tif'))
    mask_list = getFileList(os.path.join(fusar_config.MASK_ROOT, '*.tif'))
    classes = fusar_config.CLASSES_NAME
    classes_idx = fusar_config.CLASSES_INDEX
    colormap = fusar_config.CLASSES_COLORMAP

    # TRAIN CONFIG
    train_epoch = 100
    batch_size = 10
    init_lr = 1e-3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1,num_classes=fusar_config.NUM_CLASSES).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    logger.write(log_str=['lr = {}'.format(init_lr),'epoch = {}'.format(train_epoch),'device = {}'.format(device)],is_print=True)

    # DATASET CONFIG
    train_img_list, test_img_list, train_mask_list, test_mask_list = train_test_split(img_list,mask_list,test_size=0.2,random_state=12345,shuffle=True)
    train_dataset = FUSAR_DATASET(train_img_list,train_mask_list,classes,classes_idx,colormap,is_aug=True)
    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
    test_dataset = FUSAR_DATASET(test_img_list,test_mask_list,classes,classes_idx,colormap)
    test_loader = DataLoader(test_dataset,batch_size=1, shuffle=False)

    # Record the performance metrics of the best epoch and the last epoch in training
    best_overall_accuracy = 0
    best_epoch = 0

    # TRAINING AND TESTING
    segMetrics = SegMetrics(num_classes=fusar_config.NUM_CLASSES)
    for epoch_idx in range(1,train_epoch + 1):

        # Run one train epoch
        model.train()
        train_loss = 0.0
        for iter_idx, (inputs, targets) in enumerate(train_loader,start=1):
            # print("Train_Epoch:{} Iter:{}".format(epoch_idx,iter_idx))
            inputs, targets = inputs.to(device), targets.to(device)
            batch_pred = model(inputs)
            loss = loss_function(batch_pred, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            break

        # Run one test epoch
        model.eval()
        test_loss = 0.0
        segMetrics.reset()
        with torch.no_grad():
            for iter_idx, (inputs, targets) in enumerate(test_loader, start=1):
                # print("Test_Epoch:{} Iter:{}".format(epoch_idx, iter_idx))
                inputs, targets = inputs.to(device), targets.to(device)
                batch_pred = model(inputs)
                loss = loss_function(batch_pred, targets)
                test_loss += loss.item()
                segMetrics.update(preds=batch_pred.softmax(dim=1).argmax(dim=1).detach().cpu().numpy(),masks=targets.detach().cpu().numpy())
                break

        # Record the performance of the current train epoch
        current_epoch_train_loss = train_loss/len(train_loader)
        current_epoch_test_loss = test_loss/len(test_loader)
        current_epoch_segMetrics = segMetrics.getMetricsResult()
        train_log = 'EPOCH:{} train_loss:{:.4f} test_loss:{:.4f} overall accuracy:{:.4f} per-class accuracy:{} miou:{:.4f} per-class iou:{} best_OA_epoch:{}'.format(
            epoch_idx,current_epoch_train_loss,current_epoch_test_loss,
            current_epoch_segMetrics['micro_accuracy'],['{:.4f}'.format(class_accuracy) for class_accuracy in current_epoch_segMetrics['macro_accuracy']],
            current_epoch_segMetrics['macro_iou'].mean(),['{:.4f}'.format(class_iou) for class_iou in current_epoch_segMetrics['macro_iou']],
            best_epoch
        )
        logger.write(train_log,is_print=True)

        # record best epoch performance
        if current_epoch_segMetrics['micro_accuracy'] > best_overall_accuracy:
            best_overall_accuracy = current_epoch_segMetrics['micro_accuracy']
            best_epoch = epoch_idx
            logger.save_checkpoint(model,'best.pt')

        logger.save_checkpoint(model, 'last.pt')
        # One Epoch Finished
    # Training Finished




