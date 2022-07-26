'''
Note: Use centercrop(299) for inception and centercrop(224) for others in 'val'.
'''

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
import torch.utils.data as data
from PIL import Image
import os
import argparse
from tqdm import tqdm

import webdataset as wds


import json
import pandas as pd

import shutil
import h5py

model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
               'densenet121', 'densenet169', 'densenet201', 'densenet161']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Feature Extraction')


parser.add_argument('--model_name', default='resnet101')
parser.add_argument('--save_path', default='/home/pf/pfstaff/projects/andresro/infusion/data/iNaturalist/2017/')
parser.add_argument('--root_dir', default='/home/pf/pfstaff/projects/andresro/infusion/data/iNaturalist/')

parser.add_argument('--split', default='train')
parser.add_argument('--year', default='2017')
parser.add_argument('-b', '--batch_size', default=32, type=int,
    metavar='N',
    help='mini-batch size (default: 32), this is the total '
         'batch size of all GPUs on the current node when '
         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--gpu', default=0, type=int,
    help='GPU id to use.')

parser.add_argument('--subset', default='aves',choices=['aves','all'])


class Inaturalist(data.Dataset):

    def __init__(self,
                 datasplit='train',
                 year='2017',
                 args=None,
                 transform=None
                 ):
        super(Inaturalist, self).__init__()

        self.root = os.path.join(args.root_dir,year)
        self.args = args
        self.split = datasplit
        self.transform = transform

        filename = 'species_list.csv'

        df_species = pd.read_csv(filename, index_col=0)
        
        if self.split == 'train':
            json_file_path = self.root + '/train_val2017/train2017.json'
        elif self.split == 'val':
            json_file_path = self.root + '/train_val2017/val2017.json'

        with open(json_file_path) as json_file:
            annotations = json.load(json_file)
        if args.subset == 'aves':
            names_inat = [x['name'] for x in annotations['categories'] if x['supercategory'] == 'Aves']
        else:
            names_inat = [x['name'] for x in annotations['categories']]

        df1 = df_species[df_species.sci_name.isin(names_inat)]
        print(df1.shape)
        self.sci_names_inat = list(df1.sci_name)

        self.split_image_file = [x for x  in  annotations['images'] if x['file_name'].split('/')[2] in self.sci_names_inat]

    
    def __len__(self):

        return len(self.split_image_file)

    def __getitem__(self, index):

        image_path = self.split_image_file[index]['file_name']
        sci_name = image_path.split('/')[2]

        target_class = self.sci_names_inat.index(sci_name)

        image_path_full = os.path.join(self.root, image_path)
        image = Image.open(image_path_full).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, target_class, image_path


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 224

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)

    elif model_name == "resnet34":
        """ Resnet34
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)

    elif model_name == "vgg11_bn":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)

    elif model_name == "squeezenet1_0":
        """ Squeezenet1_0
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)

    elif model_name == "densenet121":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)

    elif model_name == "densenet169":
        """ Densenet169
        """
        model_ft = models.densenet169(pretrained=use_pretrained)

    elif model_name == "densenet201":
        """ Densenet201
        """
        model_ft = models.densenet201(pretrained=use_pretrained)

    elif model_name == "densenet161":
        """ Densenet161
        """
        model_ft = models.densenet161(pretrained=use_pretrained)

    elif model_name == "inception_v3":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)

        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size



def extract_features(data_loader, model, hdf5_file, is_overwrite = True):

    if not hdf5_file.endswith('.h5'):
        hdf5_file = f'{hdf5_file}.h5'

    if os.path.isfile(hdf5_file):
        if is_overwrite:
            print(' [!] Removing exsiting tar file')
            shutil.rmtree(hdf5_file, ignore_errors=True)
        else:
            raise AssertionError('file exists')
    print(hdf5_file)

    hf = h5py.File(hdf5_file,'w')

    model.eval()
    counter = 0
    with torch.no_grad():
        for input, label, image_path in tqdm(data_loader):
            output_tensor = model(input.to(device))
            output_tensor = nn.AdaptiveAvgPool2d(output_size=(1, 1))(output_tensor)

            output = output_tensor.cpu().numpy()
            output = np.squeeze(output, axis=(2, 3))
            for i in range(output.shape[0]):

                dset = hf.create_dataset(name=f'jpg/{counter}',data=output[i], compression="gzip")
                dset.attrs['file'] = image_path[i]
                dset.attrs['cls'] = label[i]
                counter+=1

    hf.close()

def main(args):
    model_name = args.model_name
    
    print("\t Working on model:", model_name)

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, use_pretrained=True)

    model_ft = model_ft.to(device)

    # Data resizing and normalization
    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    print("Initializing Datasets and Dataloaders...")

    root_dir = args.root_dir

    if args.subset == 'aves':
        name = f'{args.save_path}/inat{args.year}_aves_{args.split}_{model_name}'
    else:
        name = f'{args.save_path}/inat{args.year}_{args.split}_{model_name}'

    if args.year == '2017':
        def is_aves(x):
            return x['__key__'].split('/')[1] == 'Aves'

        if args.subset == 'all':
            def is_aves(x):
                return True

        def prepare_sample(x):

            cls_ = x['__key__'].split('/')[2]
            img = data_transforms(x['jpg'])
            return {'cls':cls_,'jpg':img}

        if args.split == 'train_val':
            dataset = (wds.WebDataset(root_dir + "/2017/train_val_images.tar.gz").decode("pil")
                        .select(predicate=is_aves)
                        .map(prepare_sample)
                        .to_tuple("jpg","cls", "__key__")
                        )
        else:
            dataset = Inaturalist(datasplit=args.split, transform=data_transforms, args = args)
    elif args.year == '2021':

        annotations_file = f'{root_dir}/2021/{args.split}.json'

        with open(annotations_file) as json_file:
            annotations = json.load(json_file)
        dirs_split = [x['image_dir_name'] for x in annotations['categories'] if x['supercategory'] == 'Birds']

        def is_aves(sample):
            return sample['__key__'].split('/')[1] in dirs_split 
        if args.subset == 'all':
            def is_aves(x):
                return True
        def prepare_sample(x):

            cls_ = x['__key__'].split('/')[1]
            img = data_transforms(x['jpg'])
            return {'cls':cls_,'jpg':img}

        tar_file = f'{root_dir}/2021/{args.split}.tar.gz'

        dataset = (wds.WebDataset(tar_file).decode("pil")
            .select(predicate=is_aves)
            .map(prepare_sample)
            .to_tuple("jpg","cls", "__key__")
            )

    dataloader = data.DataLoader(dataset, batch_size=args.batch_size,shuffle=False, drop_last=False, num_workers=8)

    model_ft = nn.Sequential(*list(model_ft.children())[:-1])
    
    extract_features(dataloader, model_ft, hdf5_file=name, is_overwrite=True)

if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")


    main(args)
