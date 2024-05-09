# -*- coding: utf-8 -*-
# @Time    : 2024/5/7 上午12:05
# @Author  : yang chen
import numpy as np
from pycm import *
# y_actu = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
# y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
#

y_true=[2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
y_pred=[0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
labels = [0,1,2]
# import os
# from Utils.ReadDataset import FUSAR_DATASET,FUSAR_DATASET_CONFIG,getFileList
# from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split
# PROJECT_HOME = os.path.abspath(r'/')
# dataset_root = os.path.join(PROJECT_HOME, 'Dataset', 'FUSAR')
# fusar_config = FUSAR_DATASET_CONFIG(dataset_root)
# img_list = getFileList(os.path.join(fusar_config.IMAGE_ROOT, '*.tif'))
# mask_list = getFileList(os.path.join(fusar_config.MASK_ROOT, '*.tif'))
# classes = fusar_config.CLASSES_NAME
# classes_idx = fusar_config.CLASSES_INDEX
# colormap = fusar_config.CLASSES_COLORMAP
# train_img_list, test_img_list, train_mask_list, test_mask_list = train_test_split(img_list, mask_list, test_size=0.2, random_state=12345, shuffle=True)
# train_dataset = FUSAR_DATASET(train_img_list, train_mask_list, classes, classes_idx, colormap)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# img,mask = train_dataset[101]
# mask = mask.numpy()
# y_true = mask.flatten()
# y_pred = mask.flatten()
cm = ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred,classes=labels)
print(' ')
print('---------Confusion Matrix-------')
print('precision_micro:',cm.PPV_Micro)
print('recall_micro:',cm.TPR_Micro)
print('Accuracy_micro:',cm.Overall_ACC) # Overall Accuracy
print('f1_micro:',cm.F1_Micro)
print('precision_macro:',cm.PPV_Macro)
print('precision_macro:',np.array(list(cm.PPV.values())).mean(),'per-class precision_macro:',list(cm.PPV.values()))
print('recall_macro:',cm.TPR_Macro)
print('recall_macro:',np.array(list(cm.TPR.values())).mean(),'per-class recall_macro:',list(cm.TPR.values()))
print('Accuracy_macro:',cm.ACC_Macro)   # Class Accuracy
print('Accuracy_macro:',np.array(list(cm.ACC.values())).mean(),'per-class precision_macro:',list(cm.ACC.values()))
print('f1_macro:',cm.F1_Macro)
print('f1_macro:',np.array(list(cm.F1.values())).mean(),'per-class recall_macro:',list(cm.F1.values()))
print('---------Scikit Learn-------')
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,jaccard_score
precision_micro = precision_score(y_true, y_pred, average='micro',labels=labels)
precision_macro = precision_score(y_true, y_pred, average='macro',labels=labels)
recall_micro = recall_score(y_true, y_pred, average='micro',labels=labels)
recall_macro = recall_score(y_true, y_pred, average='macro',labels=labels)
f1_micro = f1_score( y_true, y_pred, average='micro' ,labels=labels)
f1_macro = f1_score( y_true, y_pred, average='macro' ,labels=labels)
accuracy_micro = accuracy_score(y_true, y_pred)
# acciracy_macro # Accuracy没有官方macro的写法
precision_macro_ = []
recall_macro_ = []
f1_macro_ = []
accuracy_macro_ = []
for label in labels:
    y = [1 if y_t==label else 0 for y_t in y_true]
    p = [1 if y_p==label else 0 for y_p in y_pred]
    precision_macro_.append(precision_score(y,p))
    recall_macro_.append(recall_score(y,p))
    f1_macro_.append(f1_score(y,p))
    accuracy_macro_.append(accuracy_score(y, p))
iou_micro = jaccard_score(y_true, y_pred, average='micro')
iou_macro = jaccard_score(y_true, y_pred, average='macro')
print('precision_micro:',precision_micro)
print('recall_micro:',recall_micro)
print('accuracy_micro:',accuracy_micro)
print('f1_micro:',f1_micro)
print('precision_macro:',precision_macro)
print('precision_macro:',np.array(precision_macro_).mean(),'per-class accuracy_macro:',precision_macro_)
print('recall_macro:',recall_macro)
print('recall_macro:',np.array(recall_macro_).mean(),'per-class accuracy_macro:',recall_macro_)
print('accuracy_macro:',np.array(accuracy_macro_).mean(),'per-class accuracy_macro:',accuracy_macro_)
print('f1_macro:',f1_macro)
print('f1_macro:',np.array(f1_macro_).mean(),'per-class accuracy_macro:',f1_macro_)
print('iou_micro:',iou_micro)
print('iou_macro:',iou_macro)
print(' ')

