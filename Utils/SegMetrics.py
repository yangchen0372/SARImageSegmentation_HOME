# -*- coding: utf-8 -*-
# @Time    : 2024/5/8 下午3:59
# @Author  : yang chen
import numpy as np

### 注释都是将多分类标签转为二分类来做
class SegMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self,preds,masks):
        for pred,mask in zip(preds,masks):
            self.compute_confusion_matrix(pred.flatten(),mask.flatten())

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def compute_confusion_matrix(self, pred, mask):
        valid_mask = (mask >= 0) & (mask < self.num_classes)
        hist =np.bincount(
            self.num_classes * mask[valid_mask].astype(int) + pred[valid_mask].astype(int),
            minlength=self.num_classes**2
        ).reshape(self.num_classes,self.num_classes)
        self.confusion_matrix+=hist

    def getMicroAccuracy(self): # micro Accuracy = micro Precision = micro Recall = micro F1
        hist = self.confusion_matrix
        micro_accuracy = hist.diagonal().sum() / (hist.sum() + 1e-8)
        return micro_accuracy

    def getMacroAccuracy(self):
        hist = self.confusion_matrix    # 这段代码期待优化 #
        num_classes = self.num_classes
        macro_accuracy = []
        for class_idx in range(num_classes):
            TP = hist[class_idx,class_idx]
            FN = hist[class_idx,:].sum() - TP
            FP = hist[:,class_idx].sum() - TP
            TN = hist.sum() - TP - FN - FP
            macro_accuracy.append((TP+TN)/(TP+TN+FP+FN+1e-8))
        return np.array(macro_accuracy)

    def getMacroPrecision(self):
        hist = self.confusion_matrix
        macro_precision = hist.diagonal()/(hist.sum(axis=0) + 1e-8)
        return macro_precision

    def getMacroRecall(self):
        hist = self.confusion_matrix
        macro_recall = hist.diagonal()/(hist.sum(axis=1) + 1e-8)
        return macro_recall

    def getMacroF1(self):
        hist = self.confusion_matrix
        precision = hist.diagonal() / (hist.sum(axis=0) + 1e-8)
        recall = hist.diagonal()/(hist.sum(axis=1) + 1e-8)
        macro_f1 = 2*(precision*recall)/(precision+recall + 1e-8)
        return macro_f1

    def getMacroIoU(self):
        hist = self.confusion_matrix
        macro_iou = hist.diagonal() / (hist.sum(axis=1) + hist.sum(axis=0) - hist.diagonal() + 1e-8)
        return macro_iou

    def getMicroIoU(self):
        hist = self.confusion_matrix
        micro_iou = hist.diagonal().sum() / ((hist.sum(axis=0) + hist.sum(axis=1) -hist.diagonal()).sum() + 1e-8)
        return micro_iou

    def getMetricsResult(self):
        # micro 条件下  accuracy = precision = recall = f1
        micro_accuracy = self.getMicroAccuracy()
        micro_precision = micro_accuracy
        micro_recall = micro_accuracy
        micro_f1 = micro_accuracy

        #
        macro_precision = self.getMacroPrecision()
        macro_recall = self.getMacroRecall()
        macro_accuracy = self.getMacroAccuracy()
        macro_f1 = self.getMacroF1()

        #
        micro_iou = self.getMicroIoU()
        macro_iou = self.getMacroIoU()
        return {
            "micro_accuracy": micro_accuracy,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_accuracy": macro_accuracy,
            "macro_f1": macro_f1,
            "micro_iou": micro_iou,
            "macro_iou": macro_iou
        }

    @staticmethod
    def ShowMetricsResult(MetricsResult):
        for key,value in MetricsResult.items():
            if key.split("_")[0] == "macro":
                print(f"{key}: mean_value:{value.mean()} class_value:{value}")
            else:
                print(f"{key}: {value}")




# # debug 手动设计pred和mask
# y_true=np.array([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]).reshape(3,4)[None,:,:]
# y_pred=np.array([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]).reshape(3,4)[None,:,:]
# segMetrics = SegMetrics(num_classes=3)
# segMetrics.update(
#     y_pred,  # 最后的None用来模拟创建Batchsize
#     y_true
# )
# print('microAccuracy:',segMetrics.getMicroAccuracy())
# print('macroPrecision:',segMetrics.getMacroPrecision(),segMetrics.getMacroPrecision().mean())
# print('macroRecall:',segMetrics.getMacroRecall(),segMetrics.getMacroRecall().mean())
# print('macroAccuracy:',segMetrics.getMacroAccuracy(),segMetrics.getMacroAccuracy().mean())
# print('macroF1:',segMetrics.getMacroF1(),segMetrics.getMacroF1().mean())
# print('macroIoU:',segMetrics.getMacroIoU(), segMetrics.getMacroIoU().mean())
# print('microIoU:',segMetrics.getMicroIoU())
# SegMetrics.ShowMetricsResult(segMetrics.getMetricsResult())
#
# # debug 读取指定图片
# import os
# from Utils.ReadDataset import FUSAR_DATASET,FUSAR_DATASET_CONFIG,getFileList
# from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split
# PROJECT_HOME = os.path.abspath(r'D:\RS_WorkSpace\SARImageSegmentation_HOME')
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
# mask = mask.numpy()[None,:,:]
# segMetrics = SegMetrics(num_classes=fusar_config.NUM_CLASSES)
# segMetrics.update(
#     mask,  # 最后的None用来模拟创建Batchsize
#     mask
# )
# print('microAccuracy:',segMetrics.getMicroAccuracy())
# print('macroPrecision:',segMetrics.getMacroPrecision(),segMetrics.getMacroPrecision().mean())
# print('macroRecall:',segMetrics.getMacroRecall(),segMetrics.getMacroRecall().mean())
# print('macroAccuracy:',segMetrics.getMacroAccuracy(),segMetrics.getMacroAccuracy().mean())
# print('macroF1:',segMetrics.getMacroF1(),segMetrics.getMacroF1().mean())
# print('macroIoU:',segMetrics.getMacroIoU(), segMetrics.getMacroIoU().mean())
# print('microIoU:',segMetrics.getMicroIoU())
# SegMetrics.ShowMetricsResult(segMetrics.getMetricsResult())


