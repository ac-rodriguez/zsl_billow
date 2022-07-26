
# import h5py
import pandas as pd
import pickle
import socket
# import numpy as np
# import scipy.io as sio
import torch
# from sklearn import preprocessing
# import sys
import h5py
import os
# from logger import create_logger
# import datetime
from torch.utils.data import Dataset, TensorDataset
# import time
import os
# import string
# from utils import ResizeNoCrop
# from torchvision import transforms
# from PIL import Image
from torchvision.transforms import functional as F
import tqdm

from itertools import compress

from dataloader_billow import H5Dataset_billow_codes
from torch.utils.data import DataLoader

import json

import numpy as np

def get_basepath():

    if 'pf-pc' in socket.gethostname():
        PATH = '/home/pf/pfstaff/projects/andresro'
    else:
        PATH = '/cluster/work/igp_psr/andresro'
    return PATH



class H5Dataset_inat2017(torch.utils.data.Dataset):
    def __init__(self,opt=None,
                split= 'train',
                # transform = None,
                is_load_in_memory = False,
                ):
        super().__init__()
        self.is_load_in_memory = is_load_in_memory
        self.is_loaded = False
        
        self.opt = opt
        self.split = split
        self.base_path = get_basepath() + '/infusion/data/iNaturalist'

        self.h5_file = self.base_path+"/2017/inat2017_aves_train_val_resnet101.h5"
        # self.transform = transform

        if split == 'train':
            file_classes = f'{self.base_path}/2017/zsl_splits/seen_classes.txt'
            annotations_file = f'{self.base_path}/2017/train_val2017/train2017.json'
        elif split == 'val_seen':
            file_classes = f'{self.base_path}/2017/zsl_splits/seen_classes.txt'
            annotations_file = f'{self.base_path}/2017/train_val2017/val2017.json'
        else:
            nhop = split.split('_')[1]
            file_classes = f'{self.base_path}/2017/zsl_splits/unseen_{nhop}_classes.txt'
            annotations_file = f'{self.base_path}/2017/train_val2017/val2017.json'

        with open(annotations_file) as json_file:
            annotations = json.load(json_file)


        self.all_classes = list(pd.read_csv(f'{self.base_path}/2017/zsl_splits/all_classes.txt', header=None)[0])
        self.seen_classes = list(pd.read_csv(f'{self.base_path}/2017/zsl_splits/seen_classes.txt', header=None)[0])
        self.seen_classes_id = torch.tensor([i for i, x in enumerate(self.all_classes) if x in self.seen_classes])

        self.unseen_classes = [x for x in self.all_classes if not x in self.seen_classes]
        self.unseen_classes_id = torch.tensor([i for i, x in enumerate(self.all_classes) if x in self.unseen_classes])

        self.classes_split = list(pd.read_csv(file_classes,header=None)[0])
        self.classes_split_id = torch.tensor([i for i, x in enumerate(self.all_classes) if x in self.classes_split])

        self.index = self.read_file_list(annotations)

        self.billow_codes = H5Dataset_billow_codes(opt=opt, cls_names=self.classes_split, code_path=opt.code_path, is_load_in_memory=True)

        self.file = None

        if self.is_load_in_memory:
            
            loader = DataLoader(self, batch_size=128, shuffle=False, drop_last=False, num_workers=8)
            data_feat = []
            data_label = []
            for feat, label, _ in tqdm.tqdm(loader, desc=f'Loading inat {self.split}'):
                data_feat.extend(feat)
                data_label.extend(label)

            self.data_feat = torch.stack(data_feat)
            self.data_label = torch.stack(data_label)

            self.is_loaded = True
            self.index_perhop = None

            if split == 'val_allhop':
                classes_perhop = dict()
                classes_perhop_id = dict()
                index_perhop = dict()
                for hop in ['1hop','2hop','3hop','4hop']:
                    file_ = f'{self.base_path}/2017/zsl_splits/unseen_{hop}_classes.txt'  
                    classes_perhop[hop] = list(pd.read_csv(file_, header=None)[0])
                    classes_perhop_id[hop] = torch.tensor([i for i, x in enumerate(self.all_classes) if x in classes_perhop[hop]])
                    index_perhop[hop] = torch.tensor([i for (i,x) in enumerate(self.data_label) if x in classes_perhop_id[hop]])

                self.classes_perhop = classes_perhop
                self.classes_perhop_id = classes_perhop_id
                self.index_perhop = index_perhop                

        elif split == 'val_allhop':
            raise NotImplemented

        self.file = None


    def read_file_list(self, annotations):

        file_path = self.h5_file.replace('.h5','') + f'_file_list{self.split}.pkl'

        if os.path.isfile(file_path):
        # if False:
            with open(file_path, "rb") as fp:   # Unpickling
                sample_cls_list = pickle.load(fp)
        else:
            annotations_split = []
            for x in annotations['images']:
                sci_name = x['file_name'].split('/')[2]
                if sci_name in self.classes_split:
                    annotations_split.append(x['file_name'].replace('.jpg',''))

            self.file = h5py.File(self.h5_file, 'r')

            n_samples = len(self.file['jpg'])

            sample_cls_list = []
            for i in range(n_samples):
                sample_file = self.file['jpg'][str(i)].attrs['file']
                if sample_file in annotations_split:
                    sample_cls_list.append((i, sample_file))
            
            with open(file_path, "wb") as fp:   #Pickling
                pickle.dump(sample_cls_list, fp)
        
        self.file = None

        return sample_cls_list

    def __getitem__(self, index):

        if self.is_loaded and self.is_load_in_memory:
            sci_name = self.all_classes[self.data_label[index]]
            
            att = self.billow_codes.get_samples_class(y=sci_name, is_average = True)
            
            return self.data_feat[index], self.data_label[index], att

        i, file = self.index[index]
        
        if self.file is None:
            self.file = h5py.File(self.h5_file, 'r')

        sci_name = self.file['jpg'][str(i)].attrs['cls']

        att = self.billow_codes.get_samples_class(y=sci_name, is_average = True)

        assert sci_name == file.split('/')[2]

        x = self.file['jpg'][str(i)][:]
        target_species = self.all_classes.index(sci_name)

        return torch.from_numpy(x), torch.tensor(target_species), att


    def __len__(self):
        return len(self.index)