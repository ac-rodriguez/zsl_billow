# from typing import get_args
# from matplotlib.pyplot import get, sci
import torch
import os
import ast
from utils import ResizeNoCrop, NormalizeRBGonly, ToTensorIfNotTensor
# from utils import data_loader, ResizeNoCrop, get_basepath, GaussianBlur
# import pytorch_lightning as pl
from torchvision import transforms

from PIL import Image, ImageDraw
import pandas as pd
import pickle
import numpy as np
# import wandb
import h5py
import scipy.io as sio
import random

from dataloaders_cub import Birds

import socket
def get_basepath():

    if 'pf-pc' in socket.gethostname():
        PATH='/home/pf/pfstaff/projects/andresro'
    else:
        PATH='/cluster/work/igp_psr/andresro'
    return PATH

from torch.utils.data import DataLoader
import tqdm

def data_transforms(params, transform_type='strong'):

    # SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    if not isinstance(params,dict):
        params = params.__dict__
    std = params['std_norm']
    SetRange = transforms.Lambda(lambda X: (X-0.5)/std)
    
    # SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))
    # if params['dataset'] != 'billow_cub':
    #     raise NotImplementedError
    assert params['dataset'] in ['billow_cub','CUB','CUB_billow','CUB_dna_billow', 'inat17','inat21','inat21mini']

    if transform_type == 'basic':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                            ResizeNoCrop(size=params['img_size']),
                            transforms.ToTensor(),
                            NormalizeRBGonly([0.5, 0.5, 0.5], [std, std, std]),
                            ])
    elif transform_type == 'basic_trainmean':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                            ResizeNoCrop(size=params['img_size']),
                            transforms.ToTensor(),
                            NormalizeRBGonly(mean=(0.86641224, 0.85762388, 0.84134682), std=(0.2538723 , 0.26331718, 0.2848536))
])
    elif transform_type == 'basic_trainmean_pre':
        transform = transforms.Compose([
                            ResizeNoCrop(size=params['img_size']),
                            transforms.ToTensor(),
                            NormalizeRBGonly(mean=(0.86641224, 0.85762388, 0.84134682), std=(0.2538723 , 0.26331718, 0.2848536))
])
    elif transform_type == 'basic_trainmean_post':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
])
    elif transform_type == 'train':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                            transforms.RandomResizedCrop(size=params['img_size']),
                            ToTensorIfNotTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                            ])
    elif transform_type == 'train_pre' or transform_type == 'val_pre':
        transform = transforms.Compose([
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                            ])
    elif transform_type == 'train_post':
        transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomResizedCrop(size=params['img_size']),
                            ])
    elif transform_type == 'val':
        transform = transforms.Compose([
                            transforms.CenterCrop(size=params['img_size']),
                            ToTensorIfNotTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ])
    elif transform_type == 'val_post':
        transform = transforms.Compose([
                            transforms.CenterCrop(size=params['img_size']),
                            ])

    else:
        raise ValueError('Undefined dataset type')
    return transform




def get_embedding(file_embedding, n_classes, level_label, cls_map):

    with open(file_embedding,'rb') as file:
        embedding = pickle.load(file)
        embedding = embedding['embedding']
    
    # embedding class 

    if level_label == 'species' or level_label == 'all':
        embedding_out = torch.from_numpy(embedding).float()
    else:
        # embedding_dict = dict.fromkeys(range(self.n_classes))
        embedding_dict = {i:[] for i in range(n_classes)}
        for key, val in cls_map.items():
            
            embedding_dict[val].append(embedding[:,key])
        embedding_out = []
        for key, val in embedding_dict.items():
            vals = np.stack(val, axis=0).mean(axis=0)
            embedding_out.append(vals)

        embedding_out = np.array(embedding_out).T
        embedding_out = torch.from_numpy(embedding_out).float()
    return embedding_out

class H5Dataset_billow_cub(torch.utils.data.Dataset):
    def __init__(self,
                args =None,
                transform = None,
                transform_pre = None,
                # is_avg_species=False,
                is_flip_channels=True,
                is_use_keypoints=False,
                is_load_in_memory = False,
                ):
        super().__init__()
        self.args = args
        self.base_path = get_basepath()
        self.with_replacement = False
        self.is_use_keypoints = is_use_keypoints
        # self.is_avg_species = is_avg_species
        self.is_load_in_memory = is_load_in_memory
        self.is_loaded = False
        self.is_flip_channels = is_flip_channels
        
        self.transform = transform
        self.transform_pre = transform_pre
        self.is_use_transform_preload = False
        
        self.h5_file = self.base_path+"/birds/dataset/illustrations.h5"


        df_species = pd.read_csv(self.base_path+'/birds/dataset/species_list.csv')

        self.list_classes = list(df_species['sci_name'])

        if not 'inat' in self.args.dataset:
            df_cub = pd.read_csv(self.base_path+'/birds/CUB/classes_wikispecies_AR_orderzsl.txt',header=None, index_col=0, names =['comm_names','sci_name'])

            self.names_cub = list(df_cub['sci_name'])
            self.names_comm = list(df_cub['comm_names'])

            if self.args.dataset in ['CUB','billow_cub']:
                classes_drop = []
            elif self.args.dataset == 'CUB_billow':
                print('using the subset of 196 classes (with billow info)')
                classes_drop = np.loadtxt(self.base_path + '/birds/CUB/billow_missing_species_zsl_index.txt', dtype=int)
            elif self.args.dataset == 'CUB_dna_billow':
                print('using the subset of 191 classes (with DNA and billow info)')
                classes_drop = np.loadtxt(self.base_path + '/birds/CUB/dna_billow_missing_species_zsl_index.txt', dtype=int)            
            else:
                raise AssertionError(self.args.dataset)


            self.names_cub = [x for (i,x) in enumerate(self.names_cub) if i not in classes_drop]
            self.names_comm = [x for (i,x) in enumerate(self.names_comm) if i not in classes_drop]



            self.keypoints = pd.read_csv(f'{self.base_path}/birds/dataset/keypoints_illustrations.csv',
                                        converters={'0':ast.literal_eval,'1':ast.literal_eval,'2':ast.literal_eval,'3':ast.literal_eval},
                                        index_col=0)
        else:
            if self.args.dataset == 'inat17':
                names_inat = np.loadtxt(self.base_path+'/infusion/data/iNaturalist/2017/zsl_splits/all_classes.txt', dtype=str, delimiter='/n')         
            else:
                names_inat = np.loadtxt(self.base_path+'/infusion/data/iNaturalist/2021/zsl_splits/all_classes.txt', dtype=str, delimiter='/n')         

            self.names_cub =  list(names_inat)

            

        sample_cls_list = self.read_cls_list()

        self.index = []

        self.default_class = str(sample_cls_list.index(self.names_cub[1]))

        self.missing_classes = []

        # if self.is_avg_species:
        for i, val in enumerate(sample_cls_list):
            if val in self.names_cub:
                cls_cub = self.names_cub.index(val)
                self.index.append((cls_cub,str(i)))
        
        classes_in_dset = {x[0] for x in self.index}

        self.missing_classes = set(range(len(self.names_cub))).difference(classes_in_dset)
        for c in self.missing_classes:
            self.index.append((c, self.default_class))
        #     # for cls_cub, sci_name_cub in self.names_cub.items():
        #     for cls_cub, sci_name_cub in enumerate(self.names_cub):
        #         is_missing = True
        #         for i, val in enumerate(sample_cls_list):
        #             if val == sci_name_cub:
        #                 self.index.append([cls_cub, str(i)])
        #                 is_missing = False
        #         if is_missing:
        #             self.index.append([cls_cub, self.default_class])
        #             self.missing_classes.append(cls_cub)

        if self.is_load_in_memory:
            self.is_use_transform_preload = True
            # def collate_batch(batch):
            #     index_list = []
            #     x0_list = []
            #     for s in batch:
            #         index_list.append(s['index'])
            #         x0_list.append(s['x0'])

            #     index_list = torch.stack(index_list)
            #     return {'x0': x0_list, 'index': index_list}

            self.data = []
            self.label = []
            loader = DataLoader(self, batch_size=128, shuffle=False, drop_last=False, num_workers=8) # collate_fn=collate_batch)
            # data_feat = []
            # data_label = []
            for sample in tqdm.tqdm(loader, desc=f'Loading billow'):
                self.data.extend(sample['x0'])
                self.label.extend(sample['label'])
                # for i, index in enumerate(sample['index']):
                    # self.dict_images.update({index:sample['x0'][i]})
            self.data = torch.stack(self.data)
            self.label = torch.stack(self.label)

            self.is_loaded = True
            self.is_use_transform_preload = False

        print(len(self.index))

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

        if self.is_load_in_memory and self.is_loaded:
            x = self.data[index]
            label = self.label[index]
            if self.transform is not None:
                x = self.transform(x)
            if self.is_use_keypoints:
                raise NotImplemented

            return {'x0':x,
                    'label':label,
                    # 'file':file,
                    'index':index}

        if self.with_replacement:
            raise NotImplemented
        cls_cub , i_billow = self.index[index]

        if self.file is None:
            self.file = h5py.File(self.h5_file, 'r')

        x = self.file['jpg'][i_billow][:]
        if self.is_flip_channels:
            x = x[:,:,[2,1,0]]
        x = Image.fromarray(x)

        # rand_state = random.getstate() 
        # random.setstate(rand_state)
        if self.is_use_transform_preload:
            x = self.transform_pre(x)
        elif self.transform is not None:
            x = self.transform(x)
        
        out_dict = {'x0':x,
                'label':torch.tensor(cls_cub),
                # 'file':file_name
                'index':index}
        # if self.is_use_keypoints:
            # key_masks = self.keypoint_as_mask(x_size, i_billow, rand_state)
            # out_dict['keypoints'] = torch.cat(key_masks)

        return out_dict

    def keypoint_as_mask(self,x_size, i_billow, rand_state):

        file_name = self.file['jpg'][i_billow].attrs['file']
        keypoints = np.array(self.keypoints.loc[file_name])
        if self.is_use_keypoints:
            key_masks = []
            size_circle = int(max(10,min(x_size)*0.1)) #10 # pixels

            for (y_, x_) in keypoints:
                # creating new Image object
                key_mask = Image.new("L", x_size)

                x1,y1 = x_-size_circle,y_-size_circle
                x2,y2 = x_+size_circle,y_+size_circle
                circle = (x1,y1,x2,y2)

                # create rectangle image
                img1 = ImageDraw.Draw(key_mask)
                img1.ellipse(circle, fill=255)
                random.setstate(rand_state)
                key_mask = self.transform(key_mask)
                key_masks.append(key_mask)
        return key_masks

    def __len__(self):
        return len(self.index)



# class NumpyDataset_billow_cub(torch.utils.data.Dataset):
#     def __init__(self,
#                 args = None,
#                 transform = None,
#                 # is_avg_species=False,
#                 # is_flip_channels=True,
#                 is_use_keypoints=False,
#                 ):
#         super().__init__()
#         self.args = args
#         self.base_path = get_basepath()
#         self.with_replacement = False
#         self.is_use_keypoints = is_use_keypoints

#         self.transform_to_pil = transforms.ToPILImage(mode='RGB')
#         # self.is_avg_species = is_avg_species

#         # self.is_flip_channels = is_flip_channels

#         self.path_data = self.base_path+"/birds/dataset/illustrations_256px.npz"
#         self.data = np.load(self.path_data)

#         if transform is None:
#             self.transform = lambda x: x
#         else:
#             self.transform = transform

#         df_species = pd.read_csv(self.base_path+'/birds/dataset/species_list.csv')

#         self.list_classes = list(df_species['sci_name'])

        
#         df_cub = pd.read_csv(self.base_path+'/birds/CUB/classes_wikispecies_AR_orderzsl.txt',header=None, index_col=0, names =['comm_names','sci_name'])

#         self.names_cub = list(df_cub['sci_name'])

#         if self.args.dataset in ['CUB','billow_cub']:
#             print(' [warning no default class for missing classes, use hdf5')
#             classes_drop = []
#         elif self.args.dataset == 'CUB_billow':
#             print('using the subset of 196 classes (with billow info)')
#             classes_drop = np.loadtxt(self.base_path + '/birds/CUB/billow_missing_species_zsl_index.txt', dtype=int)
#         elif self.args.dataset == 'CUB_dna_billow':
#             print('using the subset of 191 classes (with DNA and billow info)')
#             classes_drop = np.loadtxt(self.base_path + '/birds/CUB/dna_billow_missing_species_zsl_index.txt', dtype=int)            
#         else:
#             raise AssertionError(self.args.dataset)

#         self.names_cub = [x for (i,x) in enumerate(self.names_cub) if i not in classes_drop]


#         self.keypoints = pd.read_csv(f'{self.base_path}/birds/dataset/keypoints_illustrations.csv',
#                                     converters={'0':ast.literal_eval,'1':ast.literal_eval,'2':ast.literal_eval,'3':ast.literal_eval},
#                                     index_col=0)


#         sample_cls_list = self.data['cls']

#         bool_index = [x in self.names_cub for x in sample_cls_list]

#         self.data_img = self.data['img'][bool_index]

#         self.data_label = sample_cls_list[bool_index]

#         self.classes_unique = np.unique(self.data_label)
#         self.nclasses = len(self.names_cub)

#         if len(self.classes_unique) != self.nclasses:
#             print(f'only {len(self.classes_unique)}/{self.nclasses} in billow dataset')

#         self.index = []
#         for i, sci_name in enumerate(self.data_label):
#             cls_cub = self.names_cub.index(sci_name)
#             self.index.append((cls_cub, i))
#         # self.default_class = list(sample_cls_list).index(self.names_cub[1])


#         # # for cls_cub, sci_name_cub in self.names_cub.items():
#         # for cls_cub, sci_name_cub in enumerate(self.names_cub):
#         #     is_missing = True
#         #     for i, val in enumerate(sample_cls_list):
#         #         if val == sci_name_cub:
#         #             self.index.append([cls_cub, i])
#         #             is_missing = False
#         #     if is_missing:
#         #         self.index.append([cls_cub, self.default_class])
#         #         self.missing_classes.append(cls_cub)

#         # print(len(self.index))

#         # self.index, self.labels = zip(*self.index)



#     def __getitem__(self, index):

#         cls_cub, i = self.index[index]
#         x = self.data_img[i]

#         # if self.file is None:
#         #     self.file = h5py.File(self.h5_file, 'r')
#         # if self.with_replacement:
#         #     cls_cub , i_billow = self.index_r[index]        
#         # else:
#         #     cls_cub , i_billow = self.index[index]

#         # x = self.file['jpg'][i_billow][:]
#         # if self.is_flip_channels:
#         #     x = x[:,:,[2,1,0]]
#         x = self.transform_to_pil(x) #.float()   

#         x_size = x.size

#         rand_state = random.getstate() 
#         random.setstate(rand_state)
#         x = self.transform(x)
        
#         out_dict = {'x0':x,
#                 'label':torch.tensor(cls_cub),
#                 # 'file':file_name
#                 'index':index}
                
#         if self.is_use_keypoints:
#             key_masks = self.keypoint_as_mask(x_size, i_billow, rand_state)
#             out_dict['keypoints'] = torch.cat(key_masks)

#         return out_dict

#     def keypoint_as_mask(self,x_size, i_billow, rand_state):

#         file_name = self.file['jpg'][i_billow].attrs['file']
#         keypoints = np.array(self.keypoints.loc[file_name])
#         if self.is_use_keypoints:
#             key_masks = []
#             size_circle = int(max(10,min(x_size)*0.1)) #10 # pixels

#             for (y_, x_) in keypoints:
#                 # creating new Image object
#                 key_mask = Image.new("L", x_size)

#                 x1,y1 = x_-size_circle,y_-size_circle
#                 x2,y2 = x_+size_circle,y_+size_circle
#                 circle = (x1,y1,x2,y2)

#                 # create rectangle image
#                 img1 = ImageDraw.Draw(key_mask)
#                 img1.ellipse(circle, fill=255)
#                 random.setstate(rand_state)
#                 key_mask = self.transform(key_mask)
#                 key_masks.append(key_mask)
#         return key_masks

#     def __len__(self):
#         return len(self.index)


# class NumpyDataset_billow_inat(torch.utils.data.Dataset):
#     def __init__(self,
#                 args = None,
#                 transform = None,
#                 is_use_keypoints=False,
#                 ):
#         super().__init__()
#         self.args = args
#         self.base_path = get_basepath()
#         self.with_replacement = False
#         self.is_use_keypoints = is_use_keypoints

#         if self.args.dataset == 'inat17':
#             self.year = '2017'
#         elif self.args.dataset == 'inat21':
#             self.year = '2021'
#         else:
#             raise AssertionError(f'{self.args.dataset} not implemented')

#         self.transform_to_pil = transforms.ToPILImage(mode='RGB')
#         # self.is_avg_species = is_avg_species

#         # self.is_flip_channels = is_flip_channels

#         self.path_data = self.base_path+"/birds/dataset/illustrations_256px.npz"
#         self.data = np.load(self.path_data)

#         if transform is None:
#             self.transform = lambda x: x
#         else:
#             self.transform = transform

#         df_species = pd.read_csv(self.base_path+'/birds/dataset/species_list.csv')

#         self.list_classes = list(df_species['sci_name'])

        
#         df_inat = pd.read_csv(f'{self.base_path}/infusion/data/iNaturalist/{self.year}/zsl_splits/all_classes.txt',header=None, index_col=0, names =['sci_name'])

#         self.names_inat = list(df_inat['sci_name'])

#         sample_cls_list = self.data['cls']

#         bool_index = [x in self.names_inat for x in sample_cls_list]

#         self.data_img = self.data['img'][bool_index]

#         self.data_label = sample_cls_list[bool_index]

#         self.classes_unique = np.unique(self.data_label)
#         self.nclasses = len(self.names_inat)

#         if len(self.classes_unique) != self.nclasses:
#             print(f'only {len(self.classes_unique)}/{self.nclasses} in billow dataset')

#         self.index = []
#         for i, sci_name in enumerate(self.data_label):
#             cls_cub = self.names_inat.index(sci_name)
#             self.index.append((cls_cub, i))


#     def __getitem__(self, index):

#         cls_cub, i = self.index[index]
#         x = self.data_img[i]

#         x = self.transform_to_pil(x) #.float()   

#         x_size = x.size

#         rand_state = random.getstate() 
#         random.setstate(rand_state)
#         x = self.transform(x)
        
#         out_dict = {'x0':x,
#                 'label':torch.tensor(cls_cub),
#                 # 'file':file_name
#                 'index':index}
#         # if self.is_use_keypoints:
#         #     key_masks = self.keypoint_as_mask(x_size, i_billow, rand_state)
#         #     out_dict['keypoints'] = torch.cat(key_masks)

#         return out_dict

#     def keypoint_as_mask(self,x_size, i_billow, rand_state):

#         file_name = self.file['jpg'][i_billow].attrs['file']
#         keypoints = np.array(self.keypoints.loc[file_name])
#         if self.is_use_keypoints:
#             key_masks = []
#             size_circle = int(max(10,min(x_size)*0.1)) #10 # pixels

#             for (y_, x_) in keypoints:
#                 # creating new Image object
#                 key_mask = Image.new("L", x_size)

#                 x1,y1 = x_-size_circle,y_-size_circle
#                 x2,y2 = x_+size_circle,y_+size_circle
#                 circle = (x1,y1,x2,y2)

#                 # create rectangle image
#                 img1 = ImageDraw.Draw(key_mask)
#                 img1.ellipse(circle, fill=255)
#                 random.setstate(rand_state)
#                 key_mask = self.transform(key_mask)
#                 key_masks.append(key_mask)
#         return key_masks

#     def __len__(self):
#         return len(self.index)

# class H5Dataset_billow(torch.utils.data.Dataset):
#     def __init__(self,
#                 split= 'train',
#                 level_label = 'species',
#                 dataset='billow',
#                 transform = None,
#                 n_views = 2,
#                 # h5_file = '/home/pf/pfstaff/projects/andresro/birds/codes/VQVAE2_version_12/codes.h5'
#                 ):
#         super().__init__()
#         self.base_path = get_basepath()


#         self.h5_file = self.base_path+"/birds/dataset/illustrations.h5"
#         self.level_label = level_label
#         self.transform = transform
#         self.split = split
#         self.n_views = n_views

#         df_species = pd.read_csv(self.base_path+'/birds/dataset/species_list.csv')

#         self.list_classes = list(df_species['sci_name'])
        
#         dict_levels = {'species':'sci_name',
#                        'genus':'genus',
#                        'family':'family_name',
#                        'order':'order'}
        
#         df = pd.read_csv(f'{self.base_path}/birds/dataset/splits/{self.split}_samples.txt', index_col=0)

#         self.split_list = list(df['sample'])


#         if self.level_label == 'all':

#             self.level_list = []
#             self.cls_map = []
#             self.n_classes_all = []

#             for key_name in dict_levels.values():
                
#                 # mapping from cls (always species) to level_cls (species, genus, etc.)
#                 self.level_list.append(list(dict.fromkeys(df_species[key_name])))

#                 cls_map = df_species[key_name].to_dict()
#                 self.cls_map.append({key: self.level_list[-1].index(val) for key, val in cls_map.items()})
#                 self.n_classes_all.append(len(self.level_list[-1]))

#             self.n_classes = self.n_classes_all[0]

#         else:

#             key_name = dict_levels[self.level_label]

#             # mapping from cls (always species) to level_cls (species, genus, etc.)

#             self.level_list = list(dict.fromkeys(df_species[key_name]))

#             cls_map = df_species[key_name].to_dict()
#             self.cls_map = {key: self.level_list.index(val) for key, val in cls_map.items()}
#             self.n_classes = len(self.level_list)
        
#         self.file = h5py.File(self.h5_file, 'r')

#         n_samples = len(self.file['jpg'])
        

#         file_path = os.path.dirname(self.h5_file) + '/file_list.pkl'
#         if os.path.isfile(file_path):
#             with open(file_path, "rb") as fp:   # Unpickling
#                 filename_sample_list = pickle.load(fp)
#         else:

#             filename_sample_list = []
#             for i in range(n_samples):
#                 filename_sample = self.file['jpg'][str(i)].attrs['file']
#                 try:
#                     filename_sample = filename_sample.decode('UTF-8')
#                 except (UnicodeDecodeError, AttributeError):
#                     pass
#                 filename_sample_list.append(filename_sample)
#             with open(file_path, "wb") as fp:   #Pickling
#                 pickle.dump(filename_sample_list, fp)

#         self.index = []
#         for i, filename_sample in enumerate(filename_sample_list):
#             if filename_sample in self.split_list:
#                 self.index.append(str(i))


#         # self.index = []
#         # for i in range(n_samples):
#         #     filename_sample = self.file['jpg'][str(i)].attrs['file']
#         #     try:
#         #         filename_sample = filename_sample.decode('UTF-8')
#         #     except (UnicodeDecodeError, AttributeError):
#         #         pass
#         #     if filename_sample in self.split_list:
#         #         self.index.append(str(i))


#         self.file = None
#         print(f'{split} {len(self.index)} of {n_samples}')

#         self.embedding_classes = get_embedding(
#                 file_embedding=self.base_path+'/birds/dataset/billow.unitsphere.pickle',
#                 n_classes=self.n_classes,
#                 level_label=self.level_label,
#                 cls_map=self.cls_map)
    
#     def __getitem__(self, index):

#         if self.file is None:
#             self.file = h5py.File(self.h5_file, 'r')

#         i = self.index[index]

#         sample = dict()
#         x = Image.fromarray(self.file['jpg'][i][:]) #.float()
        

#         sci_name = self.file['jpg'][i].attrs['cls']
#         target_species = self.list_classes.index(sci_name)

#         # file = self.file['jpg'][i].attrs['file']
#         # sample['file'] = file
#         sample['index'] = index

#         if self.level_label == 'all':
#             names = ['label','label0','label1','label2']
#             for cls_map, name in zip(self.cls_map, names):
#                 target = torch.tensor(cls_map[target_species])
#                 sample[name] = target

#         else:
#             target = torch.tensor(self.cls_map[target_species])
#             sample['label'] = target


#         if self.transform is not None:
#             for n in range(self.n_views):
#                 key = f'x{n}'
#                 sample[key] = self.transform(x)
#         else:
#             sample['x0'] = x
        
#         return sample
        
#     def __len__(self):
#         return len(self.index)



# class MixDataset(torch.utils.data.Dataset):
#     def __init__(self, datasets: list, type='flat'):
#         self.datasets = datasets
#         self.len_datasets = [len(d) for d in self.datasets]

#         self.index = []

#         if type =='flat':
#             for i, len_d in enumerate(self.len_datasets):
#                 for j in range(len_d):
#                     self.index.append((i,j))
#         elif type == 'balanced':
#             max_dataset = max(self.len_datasets)
#             n_reps = [ max_dataset//x for x in self.len_datasets]

#             for i, len_d in enumerate(self.len_datasets):
#                 for _ in range(n_reps[i]):
#                     for j in range(len_d):
#                         self.index.append((i,j))


#         else:
#             raise NotImplemented(type,'not implemented')

#     def __getitem__(self, i):

#         id_dataset, id = self.index[i]
#         sample =  self.datasets[id_dataset][id]
#         return sample
#         # return tuple(d[i] for d in self.datasets)

#     def __len__(self):
#         return len(self.index)


