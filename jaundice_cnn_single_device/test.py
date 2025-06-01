import os
import cv2
import glob
import json
import torch
import pickle
import random
import sklearn
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
from collections import defaultdict

from PIL import Image
from tqdm import tqdm, trange
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset, DataLoader

from utils import get_transforms, CreateModel
from utils import get_PD_result, cal_score
from utils import EyeDataset_train, EyeDataset_test
from train_test import train, valid


def set_seed(seed=308, loader=None):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_optim_crit_sched(optim, model, pos_weight, is_cls):
    if (optim == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=1e-4, amsgrad=False)
    elif (optim == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    elif (optim == 'rmsprop'):
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.256, momentum=0.9, weight_decay=1e-5)

    if (is_cls):
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]).to(device))
    else:
        criterion = torch.nn.L1Loss()
        criterion = weighted_mse_loss
        # criterion = torch.nn.MSELoss()

    # scheduler = StepLR(optimizer, step_size=3, gamma=0.97)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=4)

    return optimizer, criterion, scheduler


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':

    cuda_idx = 3

    model_name    = 'vgg16_bn'
    optim         = 'sgd'
    space         = 'rgb'
    goal          = 'cls'
    img_size      = 256
    batch_size    = 16
    img_type      = 'dng'
    data_name     = 'dataset_3_refSam2All_v1'
    ROI           = 'eye_256_256'
    skin_type     = 'with_skin'
    eval_standard = 'ba'

    data_name_map = {'samsung_s20': 'dataset_3_refSam2All_v1',
                     'iphone_11'  : 'dataset_4_refIph2All_v1',
                     'cat_s62'    : 'dataset_5_refCat2All_v1'}

    img_type_map  = {'samsung_s20': 'dng', 
                     'iphone_11'  : 'dng', 
                     'cat_s62'    : 'jpg'}

    for camera in ['samsung_s20', 'iphone_11', 'cat_s62']:
        if (camera != 'samsung_s20'):
            continue
        print('\n\n', '-------------')
       
        data_name = data_name_map[camera]
        img_type  = img_type_map[camera]
        set_seed()
        
        device   = torch.device(f'cuda:{cuda_idx}')
        patience = 30

        img_base_dir     = f'./dataset/images/{data_name}/{ROI}/{skin_type}/{img_type}'
        info_pkl_path    = f'./dataset/info_1016.pickle'
        dataset_pkl_path = f'./dataset/PD/{camera}_{img_type}.pickle'

        info_map     = pickle.load(open(info_pkl_path, 'rb'))
        dataset_dict = pickle.load(open(dataset_pkl_path, 'rb'))
        
        print(camera)
        for thr in [3, 2, 1.2]:
            if (thr != 2):
                continue

            if (goal == 'cls'):
                is_cls         = True
                pos_weight_map = {1.2: 1.2, 2: 2.4, 3: 3.5}
                pos_weight     = pos_weight_map[3]
            else:
                is_cls = False
        
            # =====================
            # Predict Testing Data
            # =====================
        
            # print('test', end='  ')
            pretrain_path = f'./result/ckpt/vgg16_bn_cls/test/{camera}_{img_type}_TB{thr}_{space}.pth' 
            model         = torch.load(pretrain_path, map_location=device)
            # train_tfms, test_tfms = get_transforms(img_size)
            # optimizer, criterion, scheduler  = get_optim_crit_sched(optim, model, pos_weight, is_cls)
            
            # test_PD       = dataset_dict['final_test_PD']
            # test_dataset  = EyeDataset_test(test_PD, img_base_dir, info_map, camera, img_type, thr, test_tfms, space, img_size, is_cls)
            # test_loader   = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)
        
            # avg_loss, score, df = valid(model, criterion, test_loader, device, is_cls)
        
            # for val in score[1:]:
            #     print(f'{val:.3f}', end='  ')
            # print()
    
