# coding = utf-8
# usr/bin/env python

'''
Author: Jantory
Email: zhang_ze_yu@outlook.com

Time: 2022/2/18 4:20 下午
Software: PyCharm
File: sanity.py
desc:
'''
from __future__ import print_function
import os
import datetime
import argparse
import itertools
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import glob
import torch as T
import torch.nn as nn
from utils import ReplayBuffer
from utils import LambdaLR
# from utils import Logger
from utils import weights_init_normal
from lib.utils import *
# training set:
from datasets_VLIR import ImageDataset
import matplotlib.pyplot as plt
from utils import mask_generator, QueueMask
from tensorboard.compat.proto.types_pb2 import DT_COMPLEX128_REF

masks_path = sorted(glob.glob(os.path.join('masks', '*.bmp')))
print(masks_path)

x = np.asarray(Image.open(masks_path[0])).astype('float')
x = transforms.ToTensor()(x)
masks = [transforms.ToTensor()(np.asarray(Image.open(p)).astype('float')) for p in masks_path]
print(masks[0].shape)