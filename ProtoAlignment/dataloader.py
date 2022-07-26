import random
import numpy as np

import torch
from torch.utils.data import Dataset
import dataloaders_billow as dl_billow
from dataloaders_cub import Birds
from dataloaders_inat import H5Dataset_inat, Webdataset_inat

# Initialization.Create_Pairs
class TrainSet_billow(Dataset):
    def __init__(self, params, is_compute_mixlist = True):

        self.params = params
        is_load_billow_in_memory = True

        if is_load_billow_in_memory:
            transform_pre = dl_billow.data_transforms(self.params,transform_type=self.params.billow_norm+'_pre')
            transform_train = dl_billow.data_transforms(self.params,transform_type=self.params.billow_norm+'_post')
        else:
            transform_pre = None
            transform_train = dl_billow.data_transforms(self.params,transform_type=self.params.billow_norm)
        try:
            assert not self.params.billow_from_numpy, 'do not use'
        except AttributeError:
            pass
        
        self.dataset_billow = dl_billow.H5Dataset_billow_cub(args=params, transform=transform_train,transform_pre=transform_pre,
        is_use_keypoints=self.params.is_pool_source_keypoints, is_load_in_memory=True)


        transform_pre = None
        transform_cub = dl_billow.data_transforms(params=self.params, transform_type='train')
            
        if not 'inat' in self.params.dataset:
            self.dataset_target = Birds(split='train', args=self.params, transform=transform_cub)
        else:
            if self.params.load_inat_in_memory:
                transform_pre = dl_billow.data_transforms(params=self.params, transform_type='train_pre')
                transform_cub = dl_billow.data_transforms(params=self.params, transform_type='train_post')

            self.dataset_target = H5Dataset_inat(split='train', args=self.params, transform=transform_cub, is_load_in_memory=self.params.load_inat_in_memory, transform_pre=transform_pre)

        print(f'train_datasets \n {self.params.dataset}: {len(self.dataset_target)} Billow: {len(self.dataset_billow)}')

        self.y_source = np.array([x[0] for x in self.dataset_billow.index])
        self.y_target = self.dataset_target.split_label

        if is_compute_mixlist:
            Training_P=[]
            Training_N=[]
            for trs in range(len(self.y_source)):
                for trt in range(len(self.y_target)):
                    if self.y_source[trs] == self.y_target[trt]:
                        Training_P.append([trs,trt, 1])
                    else:
                        Training_N.append([trs,trt, 0])
            print("Class P : ", len(Training_P), " N : ", len(Training_N))
            
            random.shuffle(Training_N)
            self.imgs = Training_P+Training_N[:3*len(Training_P)]
            random.shuffle(self.imgs)

    def __getitem__(self, idx):
        src_idx, tgt_idx, domain = self.imgs[idx]

        x_src, y_src = self.dataset_billow[src_idx]['x0'], self.y_source[src_idx]
        x_tgt, y_tgt = self.dataset_target[tgt_idx]['x0'], self.y_target[tgt_idx]

        # x_src = torch.from_numpy(x_src).unsqueeze(0)
        # x_tgt = torch.from_numpy(x_tgt).unsqueeze(0)

        return x_src, y_src, x_tgt, y_tgt

    def __len__(self):
        return len(self.imgs)



# Initialization.Create_Pairs
class TrainSet(Dataset):
    def __init__(self, domain_adaptation_task, repetition, sample_per_class):
        x_source_path = './row_data/' + domain_adaptation_task + '_X_train_source_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'
        y_source_path = './row_data/' + domain_adaptation_task + '_y_train_source_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'
        x_target_path = './row_data/' + domain_adaptation_task + '_X_train_target_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'
        y_target_path = './row_data/' + domain_adaptation_task + '_y_train_target_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'

        self.x_source=np.load(x_source_path)
        self.y_source=np.load(y_source_path)
        self.x_target=np.load(x_target_path)
        self.y_target=np.load(y_target_path)

        print("Source X : ", len(self.x_source), " Y : ", len(self.y_source))
        print("Target X : ", len(self.x_target), " Y : ", len(self.y_target))
                
        Training_P=[]
        Training_N=[]
        for trs in range(len(self.y_source)):
            for trt in range(len(self.y_target)):
                if self.y_source[trs] == self.y_target[trt]:
                    Training_P.append([trs,trt, 1])
                else:
                    Training_N.append([trs,trt, 0])
        print("Class P : ", len(Training_P), " N : ", len(Training_N))
        
        random.shuffle(Training_N)
        self.imgs = Training_P+Training_N[:3*len(Training_P)]
        random.shuffle(self.imgs)

    def __getitem__(self, idx):
        src_idx, tgt_idx, domain = self.imgs[idx]

        x_src, y_src = self.x_source[src_idx], self.y_source[src_idx]
        x_tgt, y_tgt = self.x_target[tgt_idx], self.y_target[tgt_idx]

        x_src = torch.from_numpy(x_src).unsqueeze(0)
        x_tgt = torch.from_numpy(x_tgt).unsqueeze(0)

        return x_src, y_src, x_tgt, y_tgt

    def __len__(self):
        return len(self.imgs)


class TestSet(Dataset):
    def __init__(self, domain_adaptation_task, repetition, sample_per_class):
        self.x_test = np.load('./row_data/' + domain_adaptation_task + '_X_test_target_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class)+'.npy')
        self.y_test = np.load('./row_data/' + domain_adaptation_task + '_y_test_target_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class)+'.npy')

    def __getitem__(self, idx):
        x, y = self.x_test[idx], self.y_test[idx]
        x = torch.from_numpy(x).unsqueeze(0)
        return x, y

    def __len__(self):
        return len(self.x_test)
