import os
import cv2
import glob
import csv
import copy
import torch
import pickle
import imageio
import random
import numpy as np

from torch.utils.data.dataset import Dataset

from collections import defaultdict
from torchvision import transforms
from imgaug import augmenters as iaa


class EyeDataset_test(Dataset):
    def __init__(self, PD_list, img_base_dir, info_map, camera, img_type, thr, transform, space, img_size, is_cls):
        self.PD_list = PD_list
        self.img_base_dir = img_base_dir
        self.info_map = info_map
        self.camera = camera
        self.img_type = img_type
        self.thr = thr
        self.transform = transform
        self.space = space
        self.img_size = img_size
        self.is_cls = is_cls

        self.PD_list = list(set(self.PD_list))

        # self.get_PD2VAL()
        # self.get_data_list()
        self.tbil_table = self.get_tbil_value()
        self.img_info = self.get_img_info()

        self.img_list = self.get_img_list()

    def get_img_info(self, ):
        with open('/home/ngroup/NTUH-Jiawei/data/front_face_img_info/img_name_mapping_dict.pkl', 'rb') as file:
            data = pickle.load(file)
        return data


    def get_tbil_value(self, ):
        pid_to_value = {}
        
        with open('/home/ngroup/NTUH-Jiawei/data/NTUH_patient_table_with_tbil_0608.csv', mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                pid = row['拍照流水號'].strip()
                tbil_val = row["T-bil value\u2028(收案日前)"]
                if (tbil_val.strip() == ''):
                    continue
                pid_to_value[pid] = float(tbil_val)
                pid_to_value[pid.replace('-', '_')] = float(tbil_val)
                pid_to_value[pid.replace('_', '-')] = float(tbil_val)
        # return pid_to_value
        pid_to_value = {}

        with open('/home/ngroup/NTUH-Jiawei/data/tbil.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if (row[2] == ''):
                    break
                PID = row[2]
                val = row[56]
                if (val == ''):
                    continue
                try:
                    a = float(val)
                except ValueError:
                    continue
                pid_to_value[PID] = float(val)
                pid_to_value[PID.replace('-', '_')] = float(val)
                pid_to_value[PID.replace('_', '-')] = float(val)
        return pid_to_value


    def get_img_list(self, ):
        ret = []
        img_list = glob.glob('/home/ngroup/NTUH-Jiawei/detect_FM/output/*')
        for img_path in img_list:
            img_name_ext = os.path.basename(img_path)[:-6]
            pid = self.img_info[img_name_ext]

            if (pid in self.tbil_table):
                value = self.tbil_table[pid]
            else:
                continue

            ret.append((img_path, value))
        return ret

    def get_PD2VAL(self, ):
        self.PD2TB = pickle.load(open('/project/n/jwl/tables/PD2GT/PD2TB.pickle', 'rb'))
        self.PD2CR = pickle.load(open('/project/n/jwl/tables/PD2GT/PD2CR.pickle', 'rb'))

    def get_data_list(self, ):
        self.data_list = []

        for PD in self.PD_list:
            img_list = glob.glob(f'{self.img_base_dir}/{PD}/*')

            for img_path in img_list:
                imgName_idx = os.path.basename(img_path)[:-4]
                if (imgName_idx[-2] == '_'):
                    imgName = imgName_idx[:-2]
                else:
                    imgName = imgName_idx
                imgname_ext = f'{imgName.lower()}.{self.img_type}'
                info = self.info_map[PD][imgname_ext]

                if (self.camera[:6] == 'iphone'):
                    if (info['camera_type'][:6] == 'iphone'):
                        self.data_list.append([PD, img_path])
                else:
                    if (self.camera == info['camera_type']):
                        self.data_list.append([PD, img_path])

    def __getitem__(self, index):
        img_path, value = self.img_list[index]

        '''
        img = cv2.imread(img_path)
        if (self.space == 'lab'):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        elif (self.space == 'rgb'):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transform(img)
        '''

        tfs = transforms.Compose([
                np.asarray,
                iaa.Sequential([
                    iaa.Resize({"height": self.img_size, "width": self.img_size}),
                ]).augment_image,
                np.copy,
                transforms.ToTensor()
            ])


        img = imageio.imread(img_path)
        img = tfs(img)

        # TB = self.PD2TB[PD][1]
        TB = value
        PD = 1
        data_info = [PD, img_path, TB]
        if (TB < self.thr):
            label = torch.tensor([0])
        else:
            label = torch.tensor([1])
        
        if (self.is_cls):
            return data_info, label, img
        else:
            return data_info, torch.tensor([TB / 42]), img

    def __len__(self, ):
        return len(self.img_list)


class EyeDataset_train(Dataset):
    def __init__(self, PD_list, img_base_dir, info_map, camera, img_type, thr, transform, space, img_size, is_cls):
        self.PD_list = PD_list
        self.img_base_dir = img_base_dir
        self.info_map = info_map
        self.camera = camera
        self.img_type = img_type
        self.thr = thr
        self.transform = transform
        self.space = space
        self.img_size = img_size
        self.is_cls = is_cls

        self.get_PD2VAL()
        self.get_PD2imgList()

    def get_PD2VAL(self, ):
        self.PD2TB = pickle.load(open('/project/n/jwl/tables/PD2GT/PD2TB.pickle', 'rb'))
        self.PD2CR = pickle.load(open('/project/n/jwl/tables/PD2GT/PD2CR.pickle', 'rb'))

    def get_PD2imgList(self, ):
        self.PD2imgList = defaultdict(list)

        for PD in self.PD_list:
            img_list = glob.glob(f'{self.img_base_dir}/{PD}/*')

            for img_path in img_list:
                imgName_idx = os.path.basename(img_path)[:-4]
                if (imgName_idx[-2] == '_'):
                    imgName = imgName_idx[:-2]
                else:
                    imgName = imgName_idx
                imgname_ext = f'{imgName.lower()}.{self.img_type}'
                info = self.info_map[PD][imgname_ext]

                if (self.camera[:6] == 'iphone'):
                    if (info['camera_type'][:6] == 'iphone'):
                        self.PD2imgList[PD].append(img_path)
                else:
                    if (self.camera == info['camera_type']):
                        self.PD2imgList[PD].append(img_path)

    def __getitem__(self, index):
        PD = self.PD_list[index]
        img_num = len(self.PD2imgList[PD])
        img_index = random.randint(0, img_num-1)

        '''
        img = cv2.imread(self.PD2imgList[PD][img_index])
        if (self.space == 'lab'):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        elif (self.space == 'rgb'):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        img = self.transform(img)
        '''

        tfs = transforms.Compose([
                np.asarray,
                iaa.Sequential([
                    iaa.Resize({"height": self.img_size, "width": self.img_size}),
                    # iaa.CropAndPad(percent=(-0.1, 0.1), pad_mode='edge'),
                    iaa.Rotate((-45, 45), mode='edge'),
                    iaa.TranslateX(percent=(-0.1, 0.1), mode='edge'),
                    iaa.TranslateY(percent=(-0.1, 0.1), mode='edge'),
                    iaa.Fliplr(p=0.5),
                    iaa.Flipud(p=0.5),
                    iaa.ScaleX((0.9, 1.1), mode='edge'),
                    iaa.ScaleY((0.9, 1.1), mode='edge'),
                    iaa.ShearX((-10, 10), mode='edge'),
                    iaa.ShearY((-10, 10), mode='edge'),

                    # iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
                    # iaa.GaussianBlur(sigma=(0.0, 0.1)),
                    # iaa.MultiplyBrightness(mul=(0.65, 1.35)),
                    # iaa.MultiplyAndAddToBrightness(mul=(0.9, 1.1), add=(-30, 30)),
                    # iaa.Dropout(p=(0, 0.2)),
                    iaa.Cutout(nb_iterations=(1, 3), size=0.2, squared=False),
                ]).augment_image,
                np.copy,
                transforms.ToTensor()
            ])

        img = imageio.imread(self.PD2imgList[PD][img_index])
        img = tfs(img)

        TB = self.PD2TB[PD][1]
        data_info = [PD, self.PD2imgList[PD][img_index], TB]
        if (TB < self.thr):
            label = torch.tensor([0])
        else:
            label = torch.tensor([1])

        if (self.is_cls):
            return data_info, label, img
        else:
            return data_info, torch.tensor([TB / 42]), img

    def __len__(self, ):
        return len(self.PD2imgList)
