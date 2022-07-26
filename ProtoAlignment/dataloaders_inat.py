
import torch
import os
import ast
# from torch import optim
# from torch.utils import data
# from torch.utils.tensorboard.summary import hparams
# import torchmetrics
# from models import BaseVAE
# from models.types_ import *
import json

import copy
import torch.utils.data as data
from utils import ResizeNoCrop, NormalizeRBGonly
# from utils import data_loader, ResizeNoCrop, get_basepath, GaussianBlur
# import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
# from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, IterableDataset
# import webdataset as wds
# from scheduler import CycleScheduler
from PIL import Image, ImageDraw
import pandas as pd
import pickle
import numpy as np
# import wandb
import h5py
import scipy.io as sio
import random

import tqdm
import pickle

import socket
def get_basepath():

    if 'pf-pc' in socket.gethostname():
        PATH='/home/pf/pfstaff/projects/andresro'
    else:
        PATH='/cluster/work/igp_psr/andresro'
    return PATH

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')



class H5Dataset_inat(torch.utils.data.Dataset):
    def __init__(self,
                args=None,
                split= 'train',
                transform = None,
                transform_pre = None,
                is_load_in_memory = False,
                ):
        super().__init__()
        self.is_load_in_memory = is_load_in_memory
        self.is_loaded = False
        
        self.args = args
        if args.dataset == 'inat2021mini' or args.dataset == 'inat21mini' and split == 'train':
            split = 'train_mini'

        self.split = split

        dict_years = {'inat2017':'2017',
                        'inat17':'2017',
                      'inat2021':'2021',
                      'inat21':'2021',
                      'inat2021mini':'2021',
                      'inat21mini':'2021'}

        self.year = dict_years[self.args.dataset]

        self.base_path = get_basepath() + '/infusion/data/iNaturalist'

        self.transform = transform
        self.transform_pre = transform_pre
        self.is_use_transform_preload = False
        self.transform_to_pil = transforms.ToPILImage(mode='RGB')


        self.h5_file, file_classes, annotations_file = self._get_split_files(split=split)
        
        with open(annotations_file) as json_file:
            annotations = json.load(json_file)


        self.all_classes = list(pd.read_csv(f'{self.base_path}/{self.year}/zsl_splits/all_classes.txt', header=None)[0])
        self.seen_classes = list(pd.read_csv(f'{self.base_path}/{self.year}/zsl_splits/seen_classes.txt', header=None)[0])
        self.seen_classes_id = torch.tensor([i for i, x in enumerate(self.all_classes) if x in self.seen_classes])

        self.classes_split = list(pd.read_csv(file_classes,header=None)[0])
        self.classes_split_id = torch.tensor([i for i, x in enumerate(self.all_classes) if x in self.classes_split])

        self.index = self.read_file_list(annotations)

        self.split_label = torch.tensor([self.all_classes.index(self.get_sci_name_from_filename(x[1])) for x  in  self.index])
        if split == 'val_allhop':
            classes_perhop = dict()
            classes_perhop_id = dict()
            index_perhop = dict()
            for hop in ['1hop','2hop','3hop','4hop']:
                file_ = f'{self.base_path}/{self.year}/zsl_splits/unseen_{hop}_classes.txt'  
                classes_perhop[hop] = list(pd.read_csv(file_, header=None)[0])
                classes_perhop_id[hop] = torch.tensor([i for i, x in enumerate(self.all_classes) if x in classes_perhop[hop]])
                index_perhop[hop] = torch.tensor([i for (i,x) in enumerate(self.split_label) if x in classes_perhop_id[hop]])

            self.classes_perhop = classes_perhop
            self.classes_perhop_id = classes_perhop_id
            self.index_perhop = index_perhop


        self.file = None
        if self.is_load_in_memory:

            self.is_use_transform_preload = True
            def collate_batch(batch):
                return batch
            self.data = []
            self.label = []

            n_workers = 0 # use 0 to avoid multiprocessing
            batch_size = 128
            print(f'n_workers {n_workers}, batch {batch_size} collate main process')
            
            loader = DataLoader(self, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=n_workers, collate_fn=collate_batch)
            
            for sample in tqdm.tqdm(loader, desc=f'Loading inat{self.year} {self.split}'):
                self.data.extend([x['x0'] for x in sample])
                
                self.label.extend([x['label'] for x in sample])

            self.label = torch.stack(self.label,dim=0)

            self.is_loaded = True
            self.is_use_transform_preload = False
            self.file = None
    def _get_split_files(self, split):

        if self.year == '2017':
            h5_file = self.base_path+"/2017/inat2017_aves_train_val_256px.h5"

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

        else:
        
            if split == 'train':
                h5_file = self.base_path+"/2021/inat2021_aves_train_256px.h5"
                file_classes = f'{self.base_path}/2021/zsl_splits/seen_classes.txt'
                annotations_file = f'{self.base_path}/2021/train.json'
            elif split == 'train_mini':
                h5_file = self.base_path+"/2021/inat2021_aves_train_mini_256px.h5"
                file_classes = f'{self.base_path}/2021/zsl_splits/seen_classes.txt'
                annotations_file = f'{self.base_path}/2021/train_mini.json'

            elif split == 'val_seen':
                h5_file = self.base_path+"/2021/inat2021_aves_val_256px.h5"
                file_classes = f'{self.base_path}/2021/zsl_splits/seen_classes.txt'
                annotations_file = f'{self.base_path}/2021/val.json'
            else:
                h5_file = self.base_path+"/2021/inat2021_aves_val_256px.h5"
                nhop = split.split('_')[1]
                file_classes = f'{self.base_path}/2021/zsl_splits/unseen_{nhop}_classes.txt'
                annotations_file = f'{self.base_path}/2021/val.json'
        return h5_file, file_classes, annotations_file


    def read_file_list(self, annotations):

        filename = os.path.basename(self.h5_file)
        # we reuse the file from resnet101 features, order is the same
        if self.year == '2017':
            filename_resnet101 = 'inat2017_aves_train_val_resnet101'
        else:
            if 'val' in self.split:
                filename_resnet101 = f'inat2021_aves_val_resnet101'
            else:
                filename_resnet101 = f'inat2021_aves_{self.split}_resnet101'

        file_path = self.h5_file.replace(filename,filename_resnet101) + f'_file_list{self.split}.pkl'

        if os.path.isfile(file_path):
        # if False:
            with open(file_path, "rb") as fp:   # Unpickling
                sample_cls_list = pickle.load(fp)
        else:
            annotations_split = []
            for x in tqdm.tqdm(annotations['images'],desc=f'creating pkl file {self.split}'):
                # sci_name = x['file_name'].split('/')[2]
                sci_name = self.get_sci_name_from_filename(x['file_name'])
                if sci_name in self.classes_split:
                    annotations_split.append(x['file_name'].replace('.jpg',''))

            self.file = h5py.File(self.h5_file, 'r')

            n_samples = len(self.file['jpg'])

            sample_cls_list = []
            for i in tqdm.trange(n_samples, desc='matching samples in split'):
                sample_file = self.file['jpg'][str(i)].attrs['file']
                if sample_file in annotations_split:
                    sample_cls_list.append((i, sample_file))
            
            with open(file_path, "wb") as fp:   #Pickling
                pickle.dump(sample_cls_list, fp)
        
        self.file = None

        return sample_cls_list

    def get_sci_name_from_filename(self,x):
        if self.year == '2017':
            return x.split('/')[2]
        else:
            return ' '.join(x.split('/')[1].split('_')[-2:]) 


    def __getitem__(self, index):

        if self.is_load_in_memory and self.is_loaded:
            x = self.data[index]
            label = self.label[index]
            if self.transform is not None:
                x = self.transform(x)

            return {'x0':x,'label':label,'index':index}

        i, file = self.index[index]
        

        if self.file is None:
            self.file = h5py.File(self.h5_file, 'r')

        # sci_name = self.file['jpg'][str(i)].attrs['cls']
        # assert sci_name == file.split('/')[2]
        # sci_name = file.split('/')[2]
        sci_name = self.get_sci_name_from_filename(file)
        target_species = self.all_classes.index(sci_name)

        # att = self.billow_codes.get_samples_class(y=sci_name, is_average = True)

        x = self.file['jpg'][str(i)][:]
        x = torch.tensor(x)

        if self.is_use_transform_preload:
            x = self.transform_pre(x)
        elif self.transform is not None:
            x = self.transform(x)


        return {'x0':x,
            'label':torch.tensor(target_species),
            'file':file,
            'index':index}


    def __len__(self):
        return len(self.index)


import webdataset as wds



class Webdataset_inat(IterableDataset):
    def __init__(self,
                opt=None,
                split= 'train',
                transform = None,
                # is_load_in_memory = False,
                ):
        super().__init__()
        self.base_path = get_basepath()

        self.transform = transform

        if opt.dataset == 'inat17':

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

            self.classes_split = list(pd.read_csv(file_classes,header=None)[0])
            # classes_split_id = torch.tensor([i for i, x in enumerate(all_classes) if x in classes_split])

            self.split_keys = [x['file_name'][:-4] for x  in  annotations['images'] if x['file_name'].split('/')[2] in self.classes_split]
            self.split_label = [self.all_classes.index(x.split('/')[2]) for x  in  self.split_keys]
            

            if split == 'val_allhop':
                classes_perhop = dict()
                classes_perhop_id = dict()
                index_perhop = dict()
                for hop in ['1hop','2hop','3hop','4hop']:
                    file_ = f'{self.base_path}/2017/zsl_splits/unseen_{hop}_classes.txt'  
                    classes_perhop[hop] = list(pd.read_csv(file_, header=None)[0])
                    classes_perhop_id[hop] = torch.tensor([i for i, x in enumerate(self.all_classes) if x in classes_perhop[hop]])
                    index_perhop[hop] = torch.tensor([i for (i,x) in enumerate(self.split_label) if x in classes_perhop_id[hop]])

                self.classes_perhop = classes_perhop
                self.classes_perhop_id = classes_perhop_id
                self.index_perhop = index_perhop

            # if args.split == 'train_val':
            self.dataset = (wds.WebDataset(self.base_path+"/2017/train_val_images.tar.gz").decode("pil")
                        .select(predicate=self.is_aves)
                        .select(predicate=self.is_billow)
                        .select(predicate=self.is_split)
                        .map(self.prepare_sample)
                        # .shuffle(1000)
                        # .to_tuple("jpg","cls", "__key__")
                        )

        elif self.dataset == '2021':
            pass
            # annotations_file = f'/home/pf/pfstaff/projects/andresro/infusion/data/iNaturalist/2021/{args.split}.json'

            # with open(annotations_file) as json_file:
            #     annotations = json.load(json_file)
            # dirs_split = [x['image_dir_name'] for x in annotations['categories'] if x['supercategory'] == 'Birds']

            # def is_aves(sample):
            #     return sample['__key__'].split('/')[1] in dirs_split 
            # def prepare_sample(x):

            #     cls_ = x['__key__'].split('/')[1]
            #     img = data_transforms(x['jpg'])
            #     return {'cls':cls_,'jpg':img}

            # tar_file = f'/home/pf/pfstaff/projects/andresro/infusion/data/iNaturalist/2021/{args.split}.tar.gz'

            # dataset = (wds.WebDataset(tar_file).decode("pil")
            #     .select(predicate=self.is_aves)
            #     .map(self.prepare_sample)
            #     # .to_tuple("jpg","cls", "__key__")
            #     )



    def is_aves(self,x):
        return x['__key__'].split('/')[1] == 'Aves'

    def is_billow(self,x):
        cls_ = x['__key__'].split('/')[2]
        return cls_ in self.all_classes

    def is_split(self, x):
        return x['__key__'] in self.split_keys

    def prepare_sample(self, x):

        cls_ = x['__key__'].split('/')[2]
        target_class = self.all_classes.index(cls_)
        img = self.transform(x['jpg'])
        index = self.split_keys.index(x['__key__'])

        return {'x0':img,
            'label':target_class,
            'file':x['__key__'],
            'index':index
                }

    def __iter__(self):
        return iter(self.dataset)
    def suffle(self, size):
        self.dataset = self.dataset.shuffle(size)

    def __len__(self):
        return len(self.split_keys)
