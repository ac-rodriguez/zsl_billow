

import torch
import torch.utils.data as data
from PIL import Image, ImageDraw

import numpy as np
import h5py


import socket
def get_basepath():

    if 'pf-pc' in socket.gethostname():
        PATH='/home/pf/pfstaff/projects/andresro'
    else:
        PATH='/cluster/work/igp_psr/andresro'
    return PATH




class Imagenet_subset_h5(data.Dataset):

    def __init__(self,
                 args=None,
                 transform=None,
                 ):
        super(Imagenet_subset_h5, self).__init__()

        self.base_path = get_basepath()

        self.root = self.base_path + '/infusion/data/ImageNet_train/subset_billow/'
        self.args = args
        # self.split = datasplit

        self.transform = transform

        self.h5_file = self.root + 'imagenet_subset.h5'

        self.file = h5py.File(self.h5_file, 'r')

        self.index = list(self.file['jpg'].keys())
        self.file = None
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        if self.file is None:
            self.file = h5py.File(self.h5_file, 'r')

        i = self.index[index]

        file = self.file['jpg'][i].attrs['sci_name']
        
        x = self.file['jpg'][i][:]
        x = (x * 255).astype(np.uint8)
        x = Image.fromarray(x) #.float()

        # sci_name = self.file['jpg'][i].attrs['cls']
        # target_species = self.list_classes.index(sci_name)
        target_class = torch.tensor(0)

        if self.transform is not None:
            x = self.transform(x)

        return {'x0':x,
            'label':target_class,
            'file':file}


import glob
import webdataset as wds
import tarfile


class Imagenet_subset(data.Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """

    def __init__(self,
                 args=None,
                 transform=None,
                 ):
        super(Imagenet_subset, self).__init__()

        self.base_path = get_basepath()

        self.root = self.base_path + '/infusion/data/ImageNet_train/subset_billow/'
        self.args = args
        # self.split = datasplit

        self.transform = transform

        list_tars = glob.glob(self.root+'*.tar')

        print(len(list_tars))

        self.dataset = wds.WebDataset(list_tars, shardshuffle=True).shuffle(1000).decode("pil")

        self.names = []
        for file_ in list_tars:
            tar_file = tarfile.open(file_)
            self.names.extend(tar_file.getnames())
            
            # print(n_)
        # self.n = n

        self.dataset_iter = iter(self.dataset)


    
    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """

        if index == len(self.names) - 1:
            self.dataset_iter = iter(self.dataset)

        sample = next(self.dataset_iter)

        image = sample['jpeg']
        target_class = torch.tensor(0)

        if self.transform is not None:

            image = self.transform(image)

        return {'x0':image,
            'label':target_class}
