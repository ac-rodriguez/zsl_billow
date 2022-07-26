import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
from sklearn.metrics.pairwise import cosine_similarity
from scipy.io import loadmat

import pandas as pd

import socket

from torch.utils.tensorboard.summary import text
def get_basepath():

    if 'pf-pc' in socket.gethostname():
        PATH='/home/pf/pfstaff/projects/andresro'
    else:
        PATH='/cluster/work/igp_psr/andresro'
    return PATH


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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename+'.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename+'.log', "a")
        f.write(message)  
        f.close()

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


def filter_matrices_billow(txtfile_indices, feature, label,
        trainval_loc, test_seen_loc, test_unseen_loc, text_info, original_nclasses=200):

    missing_classes = np.loadtxt( txtfile_indices, dtype=int)

    label_index = [x not in missing_classes for x in label]
    index_loc = np.argwhere(label_index).squeeze()

    labels_keep = [x for x in range(original_nclasses) if not x in missing_classes]

    feature = feature[label_index]
    label = label[label_index]
    # reindex label
    label = np.array([labels_keep.index(x) for x in label])
    if text_info is not None:
        text_info = text_info[labels_keep]

    _, x_ind, y_ind = np.intersect1d(
        trainval_loc, index_loc, return_indices=True)
    trainval_loc = y_ind[np.argsort(x_ind)]

    _, x_ind, y_ind = np.intersect1d(
        test_seen_loc, index_loc, return_indices=True)
    test_seen_loc = y_ind[np.argsort(x_ind)]

    _, x_ind, y_ind = np.intersect1d(
        test_unseen_loc, index_loc, return_indices=True)
    test_unseen_loc = y_ind[np.argsort(x_ind)]

    return feature, label, trainval_loc, test_seen_loc, test_unseen_loc, text_info

class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imageNet1K':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
                
    # not tested
    def read_h5dataset(self, opt):
        # read image feature
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".hdf5", 'r')
        feature = fid['feature'][()]
        label = fid['label'][()] 
        trainval_loc = fid['trainval_loc'][()] 
        train_loc = fid['train_loc'][()] 
        val_unseen_loc = fid['val_unseen_loc'][()] 
        test_seen_loc = fid['test_seen_loc'][()] 
        test_unseen_loc = fid['test_unseen_loc'][()] 
        fid.close()
        # read attributes
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".hdf5", 'r')
        self.attribute = fid['attribute'][()]
        fid.close()

        if not opt.validation:
            self.train_feature = feature[trainval_loc] 
            self.train_label = label[trainval_loc] 
            self.test_unseen_feature = feature[test_unseen_loc] 
            self.test_unseen_label = label[test_unseen_loc] 
            self.test_seen_feature = feature[test_seen_loc] 
            self.test_seen_label = label[test_seen_loc] 
        else:
            self.train_feature = feature[train_loc] 
            self.train_label = label[train_loc] 
            self.test_unseen_feature = feature[val_unseen_loc] 
            self.test_unseen_label = label[val_unseen_loc] 

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        self.nclasses = self.seenclasses.size(0)

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()


        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = torch.from_numpy(matcontent['w2v']).float()
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long() 
        self.test_seen_feature = torch.from_numpy(feature_val).float()
        self.test_seen_label = torch.from_numpy(label_val).long() 
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        self.test_unseen_label = torch.from_numpy(label_unseen).long() 
        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)

    def read_matdataset(self, opt):
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
                text_info, opt.normalize_embedding)
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
        self.ntest = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)        
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 
        self.Neighbours = opt.Neighbours
        # Computing the mean of the features 
        train_label_numpy = self.train_mapped_label.data.cpu().numpy()
        train_feature_numpy = self.train_feature.data.cpu().numpy()
        test_feature_numpy = self.test_unseen_feature.data.cpu().numpy()
        test_label_numpy = map_label(self.test_unseen_label, self.unseenclasses).data.cpu().numpy() 
        self.tr_cls_centroid = np.zeros([self.ntrain_class, train_feature_numpy.shape[1]]).astype(np.float32)    
        self.te_cls_centroid = np.zeros([self.ntest_class, train_feature_numpy.shape[1]]).astype(np.float32)

        for i in range(self.ntrain_class):
            self.tr_cls_centroid[i] = np.mean(train_feature_numpy[train_label_numpy == i], axis=0)

        for i in range(self.ntest_class):
            self.te_cls_centroid[i] = np.mean(test_feature_numpy[test_label_numpy == i], axis=0)
        
        train_text_feat, test_text_feat = self.text_feature_split(opt, text_info)
        self.semantic_similarity_check(opt.Neighbours, train_text_feat, test_text_feat, self.tr_cls_centroid, self.te_cls_centroid)

    def text_feature_split(self, opt, attribute):
        train_label = self.train_label.data.cpu().numpy()
        att_shape = attribute.shape[1]
        train_text_feat = np.zeros([0, att_shape])
        test_text_feat = np.zeros([0, att_shape])

        # for i in range(opt.nclass_all):
        for i in range(attribute.shape[0]):            
            if i in train_label:
                train_text_feat = np.vstack((train_text_feat, attribute[i,:]))                
            else:
                test_text_feat = np.vstack((test_text_feat, attribute[i,:]))
        
        return train_text_feat, test_text_feat

    def determine_sim_between_gen_actual(self):
        #file open 
        x = loadmat('Feature_values.mat')
        gen_feat = x['centroid_feat']
        sim_matrix = cosine_similarity(gen_feat[16,:,:], self.te_cls_centroid)
        print (sim_matrix)

    def feat_similarity(self, te_cls_centroid):
        Neighbours = 40
        tr_cls_centroid = self.tr_cls_centroid.data.cpu().numpy()
        seen_similarity_matric = cosine_similarity(te_cls_centroid, tr_cls_centroid)
        idx_mat = np.argsort(-1 * seen_similarity_matric, axis=1)
        
        semantic_similarity_seen = np.zeros((self.ntest_class, Neighbours))

        for i in range(self.ntest_class):
            for j in range(Neighbours):
                semantic_similarity_seen [i,j] = seen_similarity_matric[i, idx_mat[i,j]] 
        
        return semantic_similarity_seen, idx_mat

    def semantic_similarity_check(self, Neighbours, train_text_feature, test_text_feature, train_feature_numpy, test_feature_numpy):
        '''
        Seen Class
        '''
        seen_similarity_matric = cosine_similarity(train_text_feature, train_text_feature)
        
        #Mapping matric 
        self.idx_mat = np.argsort(-1 * seen_similarity_matric, axis=1)
        self.idx_mat = self.idx_mat[:, 0:Neighbours]

        #Neighbours Semantic similary values 
        self.semantic_similarity_seen = np.zeros((self.ntrain_class, Neighbours))

        for i in range(self.ntrain_class):
            for j in range(Neighbours):
                self.semantic_similarity_seen [i,j] = seen_similarity_matric[i, self.idx_mat[i,j]] 

        '''
        Unseen class
        '''
        unseen_similarity_matric = cosine_similarity(test_text_feature, train_text_feature)

        #Mapping matric 
        self.unseen_idx_mat = np.argsort(-1 * unseen_similarity_matric, axis=1)
        self.unseen_idx_mat = self.unseen_idx_mat[:, 0:Neighbours]

        #Neighbours Semantic similary values 
        self.semantic_similarity_unseen = np.zeros((self.ntest_class, Neighbours))
        for i in range(self.ntest_class):
            for j in range(Neighbours):
                self.semantic_similarity_unseen [i,j] = unseen_similarity_matric[i, self.unseen_idx_mat[i,j]] 

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0 
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]] 
    
    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    def next_batch_unseen_class(self, batch_size):
        idx = torch.randperm(self.ntest)[0:batch_size]
        batch_label = self.test_unseen_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_label, batch_att

    # select batch samples by randomly drawing batch_size classes    
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]
            
        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))       
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]] 
        return batch_feature, batch_label, batch_att

from dataloader_billow import H5Dataset_billow_images
from torch.utils.data import Dataset

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
        train_label_numpy = self.train_mapped_label.data.cpu().numpy()
        train_feature_numpy = self.train_feature.data.cpu().numpy()
        test_feature_numpy = self.test_unseen_feature.data.cpu().numpy()
        test_label_numpy = map_label(self.test_unseen_label, self.unseenclasses).data.cpu().numpy() 
        self.tr_cls_centroid = np.zeros([self.ntrain_class, train_feature_numpy.shape[1]]).astype(np.float32)    
        self.te_cls_centroid = np.zeros([self.ntest_class, train_feature_numpy.shape[1]]).astype(np.float32)

        for i in range(self.ntrain_class):
            self.tr_cls_centroid[i] = np.mean(train_feature_numpy[train_label_numpy == i], axis=0)

        for i in range(self.ntest_class):
            self.te_cls_centroid[i] = np.mean(test_feature_numpy[test_label_numpy == i], axis=0)


        # train_text_feat, test_text_feat = self.text_feature_split(opt, text_info)
        train_text_feat = self.attribute_seen
        test_text_feat = self.attribute_unseen
        self.semantic_similarity_check(self.opt.Neighbours, train_text_feat, test_text_feat, self.tr_cls_centroid, self.te_cls_centroid)


    def semantic_similarity_check(self, Neighbours, train_text_feature, test_text_feature, train_feature_numpy, test_feature_numpy):
        '''
        Seen Class
        '''
        seen_similarity_matric = cosine_similarity(train_text_feature, train_text_feature)
        
        #Mapping matric 
        self.idx_mat = np.argsort(-1 * seen_similarity_matric, axis=1)
        self.idx_mat = self.idx_mat[:, 0:Neighbours]

        #Neighbours Semantic similary values 
        self.semantic_similarity_seen = np.zeros((self.ntrain_class, Neighbours))

        for i in range(self.ntrain_class):
            for j in range(Neighbours):
                self.semantic_similarity_seen [i,j] = seen_similarity_matric[i, self.idx_mat[i,j]] 

        '''
        Unseen class
        '''
        unseen_similarity_matric = cosine_similarity(test_text_feature, train_text_feature)

        #Mapping matric 
        self.unseen_idx_mat = np.argsort(-1 * unseen_similarity_matric, axis=1)
        self.unseen_idx_mat = self.unseen_idx_mat[:, 0:Neighbours]

        #Neighbours Semantic similary values 
        self.semantic_similarity_unseen = np.zeros((self.ntest_class, Neighbours))
        for i in range(self.ntest_class):
            for j in range(Neighbours):
                self.semantic_similarity_unseen [i,j] = unseen_similarity_matric[i, self.unseen_idx_mat[i,j]] 

    def next_batch_unseen_class(self, batch_size):
        idx = torch.randperm(self.ntest)[0:batch_size]
        batch_label = self.test_unseen_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_label, batch_att


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


