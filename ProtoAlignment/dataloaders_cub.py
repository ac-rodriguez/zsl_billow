

import torch
import os
import torch.utils.data as data
from PIL import Image 
import numpy as np
import scipy.io as sio
import pandas as pd

import socket
def get_basepath():

    if 'pf-pc' in socket.gethostname():
        PATH='/home/pf/pfstaff/projects/andresro'
    else:
        PATH='/cluster/work/igp_psr/andresro'
    return PATH



def filter_matrices_billow(txtfile_indices, feature, label,
        trainval_loc, test_seen_loc, test_unseen_loc, text_info, original_nclasses=200):

    missing_classes = np.loadtxt( txtfile_indices, dtype=int)

    label_index = [x not in missing_classes for x in label]
    index_loc = np.argwhere(label_index).squeeze()

    labels_keep = [x for x in range(original_nclasses) if not x in missing_classes]
    if feature is not None:
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



class Birds(data.Dataset):
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
                #  root,
                 split='train',
                 args=None,
                 # cropped=False,
                #  trainvalindex=None,
                #  x_star='keypoints',
                 transform=None,
                #  target_transform=None,
                #  crop_size=(300, 300)
                 ):
        super(Birds, self).__init__()
        # self.as_pil = True
        # self.root = join(os.path.expanduser(root), self.folder)
        self.args = args
        self.base_path = get_basepath()


        if self.args.dataset in ['CUB','billow_cub']:
            # raise AssertionError('only use datasets with complete billow info')
            datafolder = 'CUB'
        elif self.args.dataset in ['CUB_billow']:
            datafolder = 'CUB'
        elif self.args.dataset == 'CUB_dna_billow':
            datafolder = 'CUB_dna'
        else:
            raise AssertionError(self.args.dataset)

        self.root = self.base_path + '/infusion/data/CUB_200_2011/CUB_200_2011/images/'
        self.args = args
        self.split = split

        self.transform = transform
        
        matcontent = sio.loadmat(f"{self.base_path}/birds/CUB/{datafolder}/res101.mat")
        # feature = matcontent['features'].T
        image_files = matcontent['image_files']

        image_files = [str(x[0][0]) for x in matcontent['image_files']]
        self.all_file = np.array([x.split('/images/')[-1] for x in image_files])

        label = matcontent['labels'].astype(int).squeeze() - 1
        
        # if opt.dataset =='CUB_dna':
            # order_split = np.loadtxt(opt.dataroot + 'zsl_dna_mapping.txt',dtype=np.int)
            
        matcontent = sio.loadmat(f"{self.base_path}/birds/CUB/{datafolder}/att_splits.mat") 
        # classes_names = [x[0][0][4:] for x in matcontent['allclasses_names']]

        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        if not self.args.filter_using_comm_names:
            if self.args.dataset == 'CUB_billow':
                _, label, trainval_loc, test_seen_loc, test_unseen_loc, _ = filter_matrices_billow(
                    self.base_path + '/birds/CUB/billow_missing_species_zsl_index.txt', None, label,
                    trainval_loc, test_seen_loc, test_unseen_loc, None, original_nclasses=200)
            elif self.args.dataset == 'CUB_dna_billow':
                _, label, trainval_loc, test_seen_loc, test_unseen_loc, _ = filter_matrices_billow(
                    self.base_path + '/birds/CUB/billow_missing_species_dna_index.txt', None, label,
                    trainval_loc, test_seen_loc, test_unseen_loc, None, original_nclasses=194)            

		# if not opt.validation:
        self.train_image_file = self.all_file[trainval_loc]
        self.test_seen_image_file = self.all_file[test_seen_loc]
        self.test_unseen_image_file = self.all_file[test_unseen_loc]

        if self.args.filter_using_comm_names:
            df_cub = pd.read_csv(self.base_path+'/birds/CUB/classes_wikispecies_AR_orderzsl.txt',header=None, index_col=0, names =['comm_names','sci_name'])

            self.comm_names_cub = list(df_cub['comm_names'])

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

            self.comm_names_cub = [x for (i,x) in enumerate(self.comm_names_cub) if i not in classes_drop]

            get_comm_name = lambda x: x.split('/')[0].split('.')[1]
            index_keep_ = [get_comm_name(x) in self.comm_names_cub for x in self.train_image_file]
            self.train_image_file = self.train_image_file[index_keep_]
            self.train_label = torch.tensor([self.comm_names_cub.index(get_comm_name(x)) for x in self.train_image_file])

            index_keep_ = [get_comm_name(x) in self.comm_names_cub for x in self.test_seen_image_file]
            self.test_seen_image_file = self.test_seen_image_file[index_keep_]
            self.test_seen_label = torch.tensor([self.comm_names_cub.index(get_comm_name(x)) for x in self.test_seen_image_file])

            index_keep_ = [get_comm_name(x) in self.comm_names_cub for x in self.test_unseen_image_file]
            self.test_unseen_image_file = self.test_unseen_image_file[index_keep_]
            self.test_unseen_label = torch.tensor([self.comm_names_cub.index(get_comm_name(x)) for x in self.test_unseen_image_file])

        else:
            raise AssertionError('use --filter_using_comm_names option')
            self.train_label = torch.from_numpy(label[trainval_loc]).long()
            self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
            self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_label.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        if self.split == 'train':
            self.split_image_file = self.train_image_file
            self.split_label = self.train_label
        elif self.split == 'unseen':
            self.split_image_file = self.test_unseen_image_file
            self.split_label = self.test_unseen_label
        elif self.split == 'seen':
            self.split_image_file = self.test_seen_image_file
            self.split_label = self.test_seen_label


    
    def __len__(self):

        return len(self.split_image_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """

        image_path = self.split_image_file[index]
        target_class = self.split_label[index]

        image_path_full = os.path.join(self.root, image_path)
        image = Image.open(image_path_full).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return {'x0':image,
            'label':target_class,
            'file':image_path,
            'index':index}