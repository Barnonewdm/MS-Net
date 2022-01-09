# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 21:35:07 2017

@author: dongming
"""

import argparse
import os
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='Unet', help='load model from [Unet FCN]')
parser.add_arugument('--data_path', type=str, help='the input data path')
parser.add_argument('--subject_num', type=int, default=1, help='subject number from 01 to 16')
parser.add_argument('--sever', type=str, default='233',help='name sever from[233,114,PC]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--model_num', type=int, default=20, help='subject number:20')
parser.add_argument('--dataset', type=str, default='NIREP', help='NIREP or LONI')

opt = parser.parse_args()

args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')
