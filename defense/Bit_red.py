import os
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels
import torch.nn as nn
from scipy.stats import norm

## Evaluation on robustness
"""
这个方法可以直接输入和输出
输出之后直接输入到各类模型中去进行测试攻击成功率
输入到哪个模型中去呢，好像只有一个数据输出，每种方法
# 如果是input transforamtion的方法， 那就输入到incv3-ens3去好了，其他两个也行，
其他的防御模型就是一个模型，可以直接输入对抗样本和输出分类结果

2022.11.18
# 一种参考variance tuning 的写法
linke:https://github.com/JHL-HUST/VT/tree/main/third_party
实验设置为:
https://github.com/JHL-HUST/VT
这个link里面给出了bit-red-reduction的设置
step_num=4, alpha=200, 攻击 base_model=Inc_v3_ens.
# 这里链接有pytorch版本的，我都不用自己写了
https://github.com/thu-ml/ares/blob/main/pytorch_ares/pytorch_ares/defense_torch/bit_depth_reduction.py
"""


# 第一个版本来自：
# link： https://github.com/ZhengyuZhao/PerC-Adversarial/blob/master/main.ipynb
# class bit_depth_red(nn.Module):
#     # levels = [7, 6, 5, 4, 3, 2]
#     def __init__(self,
#                  depth: int = 3):
#         super(bit_depth_red, self).__init__()
#         self.depth = depth
#
#     def forward(self, X_before):
#         r = 256 / (2 ** self.depth)
#         x_quan = torch.round(X_before * 255 / r) * r / 255
#         return x_quan

# 第二个版本来自：
# link:https://github.com/thu-ml/ares/blob/main/pytorch_ares/pytorch_ares/defense_torch/bit_depth_reduction.py
class BitDepthReduction(nn.Module):
    def __init__(self, compressed_bit=4):
        super(BitDepthReduction, self).__init__()
        self.compressed_bit = compressed_bit

    def forward(self, xs):
        bits = 2 ** self.compressed_bit  # 2**i
        xs_compress = (xs.detach() * bits).int()
        xs_255 = (xs_compress * (255 / bits))
        xs_compress = (xs_255 / 255)

        return xs_compress
