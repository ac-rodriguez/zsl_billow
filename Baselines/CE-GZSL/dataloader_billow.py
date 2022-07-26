# import h5py
import pandas as pd
import pickle
import socket
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import h5py
import os
from logger import create_logger
import datetime
from torch.utils.data import Dataset, TensorDataset
import time
import os
import string
# from utils import ResizeNoCrop
from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as F
import tqdm

from itertools import compress


def get_basepath():

    if 'pf-pc' in socket.gethostname():
        PATH = '/home/pf/pfstaff/projects/andresro'
    else:
        PATH = '/cluster/work/igp_psr/andresro'
    return PATH



class ResizeNoCrop(transforms.Resize):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, color=255, **kwargs):
        super().__init__(**kwargs)
        self.color = color

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """

        scale = self.size / max(img.size)

        array0 = F.resize(
            img, (int(scale * img.size[1]), int(scale * img.size[0])), self.interpolation)

        if img.mode == "RGB":
            array1 = Image.new('RGB', (self.size, self.size),
                               (self.color, self.color, self.color))
        else:
            array1 = Image.new('L', (self.size, self.size), (1,))

        array1.paste(array0, (int(
            (self.size - array0.size[0]) / 2), int((self.size - array0.size[1]) / 2)))

        return array1


class H5Dataset_billow_images(torch.utils.data.Dataset):
    def __init__(self,
                transform = None,
                is_avg_species=True,
                dataset='CUB',
                is_flip_channels=True,
                ):
        super().__init__()
        self.base_path = get_basepath()

        self.is_flip_channels = is_flip_channels
        if dataset == 'CUB':
            keep_missing = True
        elif dataset == 'CUB_billow':
            keep_missing = False
        else:
            raise NotImplementedError

        self.h5_file = self.base_path+"/birds/dataset/illustrations.h5"
        if transform is None:
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 ResizeNoCrop(size=256),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.86641224, 0.85762388, 0.84134682), std=(0.2538723, 0.26331718, 0.2848536))])
        else:
            self.transform = transform

        df_species = pd.read_csv(self.base_path+'/birds/dataset/species_list.csv')

        self.list_classes = list(df_species['sci_name'])

        
        df_cub = pd.read_csv(self.base_path + '/birds/CUB/classes_wikispecies_AR_orderzsl.txt',header=None, index_col=0, names =['comm_names','sci_name'])
        
        self.names_cub = list(df_cub['sci_name'])
        self.comm_names_cub = list(df_cub['comm_names'])

        self.file = h5py.File(self.h5_file, 'r')
       
        sample_cls_list  = self.read_cls_list()

        self.default_class = str(sample_cls_list.index(self.names_cub[1]))

        self.missing_classes = []

        self.index = []
        for cls_cub, sci_name_cub in enumerate(self.names_cub):
            is_missing = True
            for i, val in enumerate(sample_cls_list):
                if val == sci_name_cub:
                    self.index.append([cls_cub, str(i)])
                    is_missing = False
            if is_missing:
                self.index.append([cls_cub, self.default_class])
                self.missing_classes.append(cls_cub)

        print(len(self.index))
        self.index_cls = [x[0] for x in self.index]

        print('missing classes',len(self.missing_classes))

    def read_cls_list(self):

        file_path = self.h5_file.replace('.h5','_cls_list.pkl')
        if os.path.isfile(file_path):
            with open(file_path, "rb") as fp:   # Unpickling
                sample_cls_list = pickle.load(fp)
        else:

            self.file = h5py.File(self.h5_file, 'r')

            n_samples = len(self.file['jpg'])

            sample_cls_list = []
            for i in range(n_samples):
                sampe_cls = self.file['jpg'][str(i)].attrs['cls']
                try:
                    sampe_cls = sampe_cls.decode('UTF-8')
                except (UnicodeDecodeError, AttributeError):
                    pass
                sample_cls_list.append(sampe_cls)
            
            with open(file_path, "wb") as fp:   #Pickling
                pickle.dump(sample_cls_list, fp)
        
        self.file = None

        return sample_cls_list

    def __getitem__(self, index):

        if self.file is None:
            self.file = h5py.File(self.h5_file, 'r')
        cls_cub, i_billow = self.index[index]

        x = self.file['jpg'][i_billow][:]
        if self.is_flip_channels:
            x = x[:, :,[2,1,0]]
        x = Image.fromarray(x)
       
        if self.transform is not None and x is not None:
            x = self.transform(x)

        return x,torch.tensor(cls_cub)


    def __len__(self):
        return len(self.index)

    def get_samples_class(self, y, is_random = True):

        index_ = y == self.index_cls

        avail_samples = np.arange(len(self))[index_]
        if is_random:
            index_billow = np.random.permutation(avail_samples)[0]
            att, _ = self[index_billow]
            return att
        else:
            att_out = []

            for i in avail_samples:
                att, _ = self[i]
                att_out.append(att)
            return att_out


class H5Dataset_billow_codes(torch.utils.data.Dataset):
    def __init__(self,opt=None,
                cls_names = None,
                is_load_in_memory = True,
                code_path = '/scratch/andresro/leon_work/birds/codes/ResNet_mid_genus__1.0_1_version_3/billow/codes.h5',
                # transform = None,
                ):
        super().__init__()

        self.opt = opt
        self.base_path = get_basepath()
        self.cls_names = cls_names

        self.is_load_in_memory = is_load_in_memory
        self.is_loaded = False

        self.h5_file = code_path
        # self.transform = transform

        df_species = pd.read_csv(self.base_path+'/birds/dataset/species_list.csv')

        self.list_classes = list(df_species['sci_name'])
        

        self.sample_cls_list  = self.read_cls_list()


        self.index = [(x, i) for i,x in enumerate(self.sample_cls_list) if x in self.cls_names]
        self.index_cls = [x[0] for x in self.index]
        
        if self.is_load_in_memory:
            self.data = torch.FloatTensor(len(self.cls_names),self.opt.attSize)
            for i, c in enumerate(self.cls_names):
                self.data[i] = self.get_samples_class(c, is_average = True)
            self.is_loaded = True
        print('')


    def read_cls_list(self):

        file_path = self.h5_file.replace('.h5','_cls_list.pkl')
        if os.path.isfile(file_path):
            with open(file_path, "rb") as fp:   # Unpickling
                sample_cls_list = pickle.load(fp)
        else:

            self.file = h5py.File(self.h5_file, 'r')

            n_samples = len(self.file['cls'])

            sample_cls_list = []
            for i in range(n_samples):
                sampe_cls = self.file['cls'][str(i)].attrs['file'].split('/')[0]
                try:
                    sampe_cls = sampe_cls.decode('UTF-8')
                except (UnicodeDecodeError, AttributeError):
                    pass
                sample_cls_list.append(sampe_cls)
            
            with open(file_path, "wb") as fp:   #Pickling
                pickle.dump(sample_cls_list, fp)
        
        self.file = None

        return sample_cls_list

    def __getitem__(self, index):

        # if self.is_loaded:
        #     return self.data[index]
        if self.file is None:
            self.file = h5py.File(self.h5_file, 'r')
        cls_cub, i_billow = self.index[index]

        x = self.file['top'][str(i_billow)][:]

        return torch.from_numpy(x) # , torch.tensor(cls_cub)


    def __len__(self):
        return len(self.index)

    def get_samples_class(self, y, is_average = True):
        
        if self.is_loaded and self.is_load_in_memory:

            return self.data[self.cls_names.index(y)]

        assert y in self.index_cls, f'{y} not found in billow'
        avail_samples = [i for i, x in enumerate(self.index_cls) if x == y]
        if is_average:
            att_out = []

            for i in avail_samples:
                att = self[i]
                att_out.append(att)
            att_out = torch.stack(att_out).mean(dim=0)
            return att_out

        else:
            index_billow = np.random.permutation(avail_samples)[0]
            att = self[index_billow]
            return att

        # if is_random:
        #     index_billow = np.random.permutation(avail_samples)[0]
        #     att, _ = self[index_billow]
        #     return att
        # else:
        #     att_out = []

        #     for i in avail_samples:
        #         att, _ = self[i]
        #         att_out.append(att)
        #     return att_out