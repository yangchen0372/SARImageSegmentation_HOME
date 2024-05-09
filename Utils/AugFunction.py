# -*- coding: utf-8 -*-
# @Time    : 2024/4/29 下午10:38
# @Author  : yang chen
import random

import cv2
import numpy as np
import torch


def to_tensor(img, is_mask=False):
    '''
    将ndarray格式的image转为符合pytorch的格式
    :param img:
    :return:
    '''
    if not is_mask:  # 对mask不增加维度
        if img.ndim == 2:
            img = img[:, :, None]
        img = img.transpose((2, 0, 1))  # [H,W,C] -> [H,W]
        img = torch.from_numpy(img).contiguous()
    else:  # return mask
        img = torch.from_numpy(img)
        img = img.long()
        return img
    if isinstance(img, torch.ByteTensor):  #
        return img.float().div(255)
    else:
        return img


def rotate(img, angle, interpolation=1, borderValue=0):
    """
    旋转一副图像(会裁剪)
    :param img:符合Opencv读入格式.
    :param angle:旋转的角度
    :param interpolation:插值方式
           INTER_NEAREST = 0
           INTER_LINEAR = 1
           INTER_CUBIC = 2
           INTER_AREA = 3
            !!interpolation还可以选为WARP_INVERSE_MAP=16,可以进行逆操作.!!
    :param borderValue:边界值,注意对于彩色图边界值应该给的是数组(R,G,B),只给255代表红色(255,0,0)
    """
    img_size_h, img_size_w = img.shape[:2]  # 获取图像尺寸
    cx, cy = int(img_size_w / 2), int(img_size_h / 2)  # 获得图像中心坐标
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)  # 获得变换矩阵
    img = cv2.warpAffine(
        img,
        M,
        (img_size_w, img_size_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=borderValue
    )
    return img


def scale(img, mask, factor=1):
    """
    按比例放大或缩小图像.
    :param img: color-RGB-[H,W,C]
    :param mask:grayscale-[H,W]
    :param factor: 缩放因子,值范围建议(不强制)在(0,2~]
           1为返回原图
    """
    if factor == 1:
        return img, mask
    elif factor <= 0:
        raise ValueError('scale_factor != 0')

    h, w = img.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)

    # 缩放图像
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # padding or crop
    x, y = int(np.random.uniform(0, abs(new_w - w))), int(np.random.uniform(0, abs(new_h - h)))

    if factor < 1.0:
        # padding
        # img_padding = np.zeros([h, w, 3], dtype=np.uint8) # RGB
        # img_padding[y:y + new_h, x:x + new_w, :] = img
        img_padding = np.zeros([h, w], dtype=np.uint8)  # Gray
        img_padding[y:y + new_h, x:x + new_w] = img
        mask_padding = np.zeros([h, w], dtype=np.uint8)
        mask_padding[y:y + new_h, x:x + new_w] = mask
        return img_padding, mask_padding
    elif factor > 1.0:
        # crop
        # img_crop = img[y:y + h, x:x + w, :] # RGB
        img_crop = img[y:y + h, x:x + w]  # Gray
        mask_crop = mask[y:y + h, x:x + w]
        return img_crop, mask_crop


class AugFunction():

    @staticmethod
    def RandomRotate(img_mask):
        img = img_mask[0]
        mask = img_mask[1]
        angle = 360.0
        rotate_angle = angle * random.random()
        img = rotate(img, rotate_angle, interpolation=1)
        mask = rotate(mask, rotate_angle, interpolation=0)
        return img, mask

    @staticmethod
    def RandomVerticalFlip(img_mask, p=0.5):
        img = img_mask[0]
        mask = img_mask[1]
        if random.random() > p:
            return img, mask
        else:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
            return img, mask

    @staticmethod
    def RandomHorizontalFlip(img_mask, p=0.5):
        img = img_mask[0]
        mask = img_mask[1]
        if random.random() > p:
            return img, mask
        else:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
            return img, mask

    @staticmethod
    def RandomScale(img_mask, p=0.5, scales=[0.75, 0.85, 0.95, 1, 1.05, 1.15, 1.25]):
        img = img_mask[0]
        mask = img_mask[1]
        if random.random() > p:
            return img, mask
        random_scale = random.choice(scales)
        img, mask = scale(img, mask, random_scale)
        return img, mask
