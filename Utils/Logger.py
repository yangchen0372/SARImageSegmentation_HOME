# -*- coding: utf-8 -*-
# @Time    : 2024/5/6 下午8:08
# @Author  : yang chen
import os
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, log_dir):
        self.root_path = log_dir
        self.writer = SummaryWriter(log_dir=self.root_path)
        self.log_path = os.path.join(self.root_path, 'log.txt')

    def save_train_loss(self, value, num_cv, epoch):
        title = 'train_loss/kfold_{}'.format(num_cv)
        self.writer.add_scalar(title, value, epoch)

    def save_test_loss(self, value, num_cv, epoch):
        title = 'test_loss/kfold_{}'.format(num_cv)
        self.writer.add_scalar(title, value, epoch)

    def save_test_metrics(self, metrics_name, value, num_cv, epoch):
        title = 'metrics_{}/K_fold_{}'.format(metrics_name, num_cv)
        self.writer.add_scalar(title, value, epoch)

    def write(self, log_str , is_print=False):
        f = open(self.log_path, 'a')
        if isinstance(log_str, str):
            f.write(str(log_str) + '\n')
            if is_print:
                print(log_str)
        elif isinstance(log_str, list):
            for line_str in log_str:
                f.write(line_str + '\n')
                if is_print:
                    print(line_str)
        f.close()

    def save_checkpoint(self, parameter ,file_name):
        pass
        # torch.save({})


    def close(self):
        self.writer.close()