import argparse
import pydoc
import random
import sys
from importlib import import_module
from pathlib import Path  # os.path.join的进阶版
from typing import Union  # 提示输入类型的包

import numpy as np
import torch
from addict import Dict
from torch import nn


class two_sum():
    def __init__(self, a):
        self.a = a + 1


class three_sum():
    def __init__(self, name, age):
        self.dit = name

    def print_sum(self):
        print(self.dit)


def dictionary(**kwargs):
    for key, value in kwargs.items():
        print(key, value)


class ConfigDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        else:
            return value
        raise ex

    def _generate_matrix(gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < 2)
        label = 2 * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=2 ** 2)
        confusion_matrix = count.reshape(2, 2)
        return confusion_matrix

if __name__ == '__main__':
    img = np.ones([5, 5], dtype=int)
    label = np.zeros([5, 5])
    for i in range(25):
        label[i // 5, i % 5] = random.randint(-5, 5)
    print(img)
    print(label)
    label = torch.Tensor(label)
    pre_mask = nn.Softmax(dim=1)(label)
    print(pre_mask)


    # parser = argparse.ArgumentParser(description="Demo of argparse")
    # parser.add_argument('-n', '--name', default='Kate')
    # parser.add_argument('-l', '--log_name')
    # args = parser.parse_args()
    # name = args.name
    # log_name = 'inriabuilding/{}'.format(args.log_name)
    # print('name:%s' % name)
    # print('log_name:%s' % log_name)
