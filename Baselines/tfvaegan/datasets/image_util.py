#import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import pandas as pd
from torch.utils.data import Dataset


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label


def normalize_billow(text_info, option):
    if option == 'mean_var':
        if len(text_info.shape) == 2:
            text_info = text_info[...,np.newaxis,np.newaxis]
        mean, var = text_info.mean(axis=(0,2,3), keepdims=True), text_info.std(axis=(0,2,3),keepdims=True)
        text_info = (text_info- mean) / var
    elif option == 'exp':
        text_info = np.exp(text_info)
    elif option == 'max':
        text_info = text_info / text_info.max()
    elif option == 'l2':
        text_info / np.linalg.norm(text_info,axis=-1)[:,np.newaxis]

    text_info = text_info.reshape(200,-1)
    return text_info


class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):
        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        # feature = matcontent['features'].T
        # label = matcontent['labels'].astype(int).squeeze() - 1
        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        # train_loc = matcontent['train_loc'].squeeze() - 1
        # val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        # test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        # test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1    

        # self.attribute = torch.from_numpy(matcontent['att'].T).float()
        # self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),self.attribute.size(1))

        datafolder = 'CUB'
        matcontent = sio.loadmat(
            opt.dataroot + "/" + datafolder + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        image_files = [str(x[0][0]) for x in matcontent['image_files']]
        self.all_file = np.array([x.split('/images/')[-1] for x in image_files])


        label = matcontent['labels'].astype(int).squeeze() - 1

        classes_text_info = 'default_zsl_index'
        
        matcontent = sio.loadmat(
            opt.dataroot + "/" + datafolder + "/att_splits.mat")

        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        if opt.class_embedding == 'att':
            text_info = matcontent['att'].T
        # elif opt.class_embedding in ['att', 'att_dna', 'att_w2v']:
        elif opt.class_embedding in ['att_dna', 'att_w2v']:
            matcontent = sio.loadmat(opt.dataroot + "/CUB_dna/att_splits.mat")
            text_info = matcontent[opt.class_embedding].T
            classes_text_info = [x[0][0][4:] for x in matcontent['allclasses_names']]
            assert 'dna' in opt.dataset, f'class_embedding {opt.class_embedding} not complete for {opt.dataset}'
        elif opt.class_embedding == 'billow':

            text_info = np.load(opt.code_path)['tops']
            text_info = normalize_billow(
                text_info, 'l2')
            # classes_text_info = 'default_zsl_index'
            # if datafolder == 'CUB_dna':
            #     text_info = text_info[order_split]

        elif opt.class_embedding == 'sent':
            matcontent = sio.loadmat(opt.dataroot + "/CE-GZSL/sent_splits.mat")
            # classes_text_info = 'default_zsl_index'
            text_info = matcontent['att'].T
            # if datafolder == 'CUB_dna':
            #     text_info = text_info[order_split]
            # else:
            #     assert np.array_equal(
            #         trainval_loc, matcontent['trainval_loc'].squeeze() - 1)

        assert opt.filter_using_comm_names

        if not opt.validation:

            self.train_image_file = self.all_file[trainval_loc]
            self.test_seen_image_file = self.all_file[test_seen_loc]
            self.test_unseen_image_file = self.all_file[test_unseen_loc]

            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
                self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long() 


        if opt.filter_using_comm_names:
            df_cub = pd.read_csv(opt.dataroot+'/classes_wikispecies_AR_orderzsl.txt',header=None, index_col=0, names =['comm_names','sci_name'])

            self.comm_names_zsl_order = list(df_cub['comm_names'])

            if opt.dataset in ['CUB','billow_cub']:
                classes_drop = []
            elif opt.dataset == 'CUB_billow':
                print('using the subset of 196 classes (with billow info)')
                classes_drop = np.loadtxt(opt.dataroot + '/billow_missing_species_zsl_index.txt', dtype=int)
            elif opt.dataset == 'CUB_dna_billow':
                print('using the subset of 191 classes (with DNA and billow info)')
                classes_drop = np.loadtxt(opt.dataroot + '/dna_billow_missing_species_zsl_index.txt', dtype=int)            
            else:
                raise AssertionError(opt.dataset)

            self.comm_names_cub = [x for (i,x) in enumerate(self.comm_names_zsl_order) if i not in classes_drop]

            if classes_text_info == 'default_zsl_index':
                order_split = [self.comm_names_zsl_order.index(x) for x in self.comm_names_cub]
                # index_text_info = [x in self.comm_names_cub for x in list(df_cub['comm_names'])]
            else:
                order_split = [classes_text_info.index(x) for x in self.comm_names_cub]

                # index_text_info = [x in self.comm_names_cub for x in classes_text_info]
            text_info = text_info[order_split]

            get_comm_name = lambda x: x.split('/')[0].split('.')[1]
            index_keep_ = [get_comm_name(x) in self.comm_names_cub for x in self.train_image_file]
            self.train_image_file = self.train_image_file[index_keep_]
            self.train_label = torch.tensor([self.comm_names_cub.index(get_comm_name(x)) for x in self.train_image_file])
            self.train_feature = self.train_feature[index_keep_]
            

            index_keep_ = [get_comm_name(x) in self.comm_names_cub for x in self.test_seen_image_file]
            self.test_seen_image_file = self.test_seen_image_file[index_keep_]
            self.test_seen_label = torch.tensor([self.comm_names_cub.index(get_comm_name(x)) for x in self.test_seen_image_file])
            self.test_seen_feature = self.test_seen_feature[index_keep_]

            index_keep_ = [get_comm_name(x) in self.comm_names_cub for x in self.test_unseen_image_file]
            self.test_unseen_image_file = self.test_unseen_image_file[index_keep_]
            self.test_unseen_label = torch.tensor([self.comm_names_cub.index(get_comm_name(x)) for x in self.test_unseen_image_file])
            self.test_unseen_feature = self.test_unseen_feature[index_keep_]

        if text_info is not None:
            self.attribute = torch.from_numpy(text_info).float()
        else:
            self.attribute = text_info



        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))


        self.ntrain = self.train_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 

    def next_seen_batch(self, seen_batch):
        idx = torch.randperm(self.ntrain)[0:seen_batch]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_att




import time
import os
import string

def add_version_path(out_dir, timestamp=False, is_letter = False):

    ''' Add a letter to the output directory of the summaries to avoid overwriting if several jobs are run at the same time'''

    if timestamp:
        out_dir = out_dir + time.strftime("%y%m%d_%H%M", time.gmtime())
    i = 0
    letter = '0'

    if is_letter:
        list_versions = string.ascii_lowercase
    else:
        list_versions = [str(x) for x in range(1,1000)]
    created = False

    while not created:
        if not os.path.exists(out_dir + letter):
            try:
                os.makedirs(out_dir + letter)
                created = True
            except:
                pass
        # Check if the folder contains any kind of file
        elif len([name for name in os.listdir(out_dir + letter) if os.path.isfile(os.path.join(out_dir + letter, name))]) == 0:
            created = True
        else:
            letter = list_versions[i]
            i += 1
    return out_dir + letter




class Data_empty(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self,key, val)
        self.ntrain = self.train_feature.size()[0]
        self.ntest = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)        
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 


class MyDataset(Dataset):
    def __init__(self, X, Y, att, opt):
        super().__init__()
        self.opt = opt

        self.X = X
        self.Y = Y
        self.att = att

    def __getitem__(self, index):

        x, y = self.X[index], self.Y[index]
        
        if self.att is None:
            att = self.data_billow.get_samples_class(y.numpy(),is_random=True)

        else:
            att = self.att[y]
            #ich liebe dich

        return x, y, att        

    def __len__(self):
        return self.X.size(0)


