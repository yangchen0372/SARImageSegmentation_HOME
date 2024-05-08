# -*- coding: utf-8 -*-
# @Time    : 2024/4/29 下午10:38
# @Author  : yang chen
import torch
def to_tensor(img,is_mask=False):
    '''
    将ndarray格式的image转为符合pytorch的格式
    :param img:
    :return:
    '''
    if not is_mask: # 对mask不增加维度
        if img.ndim == 2:
            img = img[:, :, None]
        img = img.transpose((2, 0, 1))   # [H,W,C] -> [H,W]
        img = torch.from_numpy(img).contiguous()
    else:   # return mask
        img = torch.from_numpy(img)
        img = img.long()
        return img
    if isinstance(img, torch.ByteTensor):   #
        return img.float().div(255)
    else:
        return img