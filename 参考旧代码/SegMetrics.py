# -*- coding: utf-8 -*-
# @Time    : 2024/5/8 下午3:59
# @Author  : yang chen
import numpy as np

### 注释都是将多分类标签转为二分类来做
class SegMetrics:
    """

    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.classes_confusion_matrix = [np.zeros((2, 2)) for _ in range(self.num_classes)]

    def update(self,preds,masks):
        for pred,mask in zip(preds,masks):
            self.compute_confusion_matrix(pred.flatten(),mask.flatten())

    def compute_confusion_matrix(self, pred, mask):
        valid_mask = (mask >= 0) & (mask < self.num_classes)
        hist =np.bincount(
            self.num_classes * mask[valid_mask].astype(int) + pred[valid_mask].astype(int),
            minlength=self.num_classes**2
        ).reshape(self.num_classes,self.num_classes)
        self.confusion_matrix+=hist

        # # classes_confusion_matrix
        # for classes_idx, label in enumerate(range(self.num_classes)):
        #     #
        #     y = np.array([1 if y_t == label else 0 for y_t in mask])
        #     p = np.array([1 if y_p == label else 0 for y_p in pred])
        #     #
        #     valid_mask = (y >= 0) & (y < 2)
        #     hist = np.bincount(
        #         2 * y[valid_mask].astype(int) + p[valid_mask].astype(int),
        #         minlength=2 ** 2
        #     ).reshape(2, 2)
        #     self.classes_confusion_matrix[classes_idx] += hist


    def getMicroAccuracy(self): # micro Accuracy = micro Precision = micro Recall = micro F1
        hist = self.confusion_matrix
        micro_accuracy = hist.diagonal().sum() / hist.sum()
        return micro_accuracy

    def getMacroAccuracy(self):
        # classes_hist = self.classes_confusion_matrix
        # macro_accuracy = []
        # for idx, class_hist in enumerate(classes_hist):
        #     macro_accuracy.append(class_hist.diagonal().sum() / class_hist.sum())
        hist = self.confusion_matrix
        num_classes = self.num_classes
        macro_accuracy = []
        for class_idx in range(num_classes):
            TP = hist[class_idx,class_idx]
            FN = hist[class_idx,:].sum() - TP
            FP = hist[:,class_idx].sum() - TP
            TN = hist.sum() - TP - FN - FP
            macro_accuracy.append((TP+TN)/(TP+TN+FP+FN))
        return np.array(macro_accuracy)

    def getMacroPrecision(self):
        # classes_hist = self.classes_confusion_matrix
        # macro_precision = []
        # for class_idx, class_hist in enumerate(classes_hist):
        #     macro_precision.append(class_hist[1,1] / class_hist[:,1].sum())
        hist = self.confusion_matrix
        macro_precision = hist.diagonal()/hist.sum(axis=0)
        return macro_precision

    def getMacroRecall(self):
        # classes_hist = self.classes_confusion_matrix
        # macro_recall = []
        # for class_idx, class_hist in enumerate(classes_hist):
        #     macro_recall.append(class_hist[1,1] / class_hist[1,:].sum())
        hist = self.confusion_matrix
        macro_recall = hist.diagonal()/hist.sum(axis=1)
        return macro_recall

    def getMacroF1(self):
        # classes_hist = self.classes_confusion_matrix
        # macro_f1 = []
        # for class_idx, class_hist in enumerate(classes_hist):
        #     precision = class_hist[1,1] / class_hist[:,1].sum()
        #     recall = class_hist[1,1] / class_hist[1,:].sum()
        #     macro_f1.append((2*precision*recall)/(precision+recall))
        hist = self.confusion_matrix
        precision = hist.diagonal() / hist.sum(axis=0)
        recall = hist.diagonal()/hist.sum(axis=1)
        macro_f1 = 2*(precision*recall)/(precision+recall)
        return macro_f1

    def getMacroIoU(self):
        hist = self.confusion_matrix
        macro_iou = hist.diagonal() / (hist.sum(axis=1) + hist.sum(axis=0) - hist.diagonal())
        return macro_iou

    def getMicroIoU(self):
        hist = self.confusion_matrix
        micro_iou = hist.diagonal().sum() / (hist.sum(axis=0) + hist.sum(axis=1) -hist.diagonal()).sum()
        return micro_iou




y_true=[2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
y_pred=[0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
import os
from Utils.ReadDataset import FUSAR_DATASET,FUSAR_DATASET_CONFIG,getFileList
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
PROJECT_HOME = os.path.abspath(r'/')
dataset_root = os.path.join(PROJECT_HOME, 'Dataset', 'FUSAR')
fusar_config = FUSAR_DATASET_CONFIG(dataset_root)
img_list = getFileList(os.path.join(fusar_config.IMAGE_ROOT, '*.tif'))
mask_list = getFileList(os.path.join(fusar_config.MASK_ROOT, '*.tif'))
classes = fusar_config.CLASSES_NAME
classes_idx = fusar_config.CLASSES_INDEX
colormap = fusar_config.CLASSES_COLORMAP
train_img_list, test_img_list, train_mask_list, test_mask_list = train_test_split(img_list, mask_list, test_size=0.2, random_state=12345, shuffle=True)
train_dataset = FUSAR_DATASET(train_img_list, train_mask_list, classes, classes_idx, colormap)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
img,mask = train_dataset[101]
mask = mask.numpy()
segMetrics = SegMetrics(num_classes=4)
segMetrics.update(
    mask[None,:,:],  # 最后的None用来模拟创建Batchsize
    mask[None,:,:]
)
print('microAccuracy:',segMetrics.getMicroAccuracy())
print('macroPrecision:',segMetrics.getMacroPrecision(),segMetrics.getMacroPrecision().mean())
print('macroRecall:',segMetrics.getMacroRecall(),segMetrics.getMacroRecall().mean())
print('macroAccuracy:',segMetrics.getMacroAccuracy(),segMetrics.getMacroAccuracy().mean())
print('macroF1:',segMetrics.getMacroF1(),segMetrics.getMacroF1().mean())
print('macroIoU:',segMetrics.getMacroIoU(), segMetrics.getMacroIoU().mean())
print('microIoU:',segMetrics.getMicroIoU())

