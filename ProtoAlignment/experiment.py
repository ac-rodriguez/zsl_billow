from multiprocessing import sharedctypes
from idna import valid_contextj
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import os
from dotmap import DotMap
import copy
import torchutils
from models import CosineClassifier, MemoryBank, SSDALossModule
from model import Network, Normalize
import dataloaders_billow

from dataloaders_billow import Birds
from dataloaders_inat import H5Dataset_inat
import torchmetrics
import numpy as np
from dataloader import TrainSet_billow
from utils import AverageMeter

from torch.utils.data import DataLoader, RandomSampler


class ExperimentMemoryBank(object):
    def __init__(self, config, logger = None) -> None:
        self.config = config
        self.out_dim = 512

        self.dataset_classes = {'CUB_billow':196,
                    'CUB_dna_billow': 191,
                    'CUB':200,
                    'inat17':895,
                    'inat21':1485,
                    'inat21mini':1485}

        self.num_class = self.dataset_classes[self.config.dataset]
        self.temp = 0.1
        self.device = 'cuda'
        self.cls = True
        self.logger = logger
        self.gen_unseen_centroid = False
        self.copy_checkpoint_freq = self.config.copy_checkpoint_freq

        self.loss_fn = SSDALossModule(config)

        self.model = Network(self.config, out_dim=self.out_dim)

        torchutils.weights_init(self.model.fc)

        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss().cuda()
        cls_head = CosineClassifier(
            num_class=self.num_class, inc=self.out_dim, temp=self.temp
        )
        torchutils.weights_init(cls_head)
        self.cls_head = cls_head.cuda()

        if self.config.unseen_centroid_on_target > 0:
            self.gen_unseen_centroid = True
            if self.config.unseen_prediction == 'gan-gp':
                self.nz = 512
                self.netG = nn.Sequential(nn.Linear(self.out_dim+self.nz,self.out_dim),
                                                    nn.LeakyReLU(0.2,inplace=True),
                                                    nn.Linear(self.out_dim,self.out_dim)
                                                    ).cuda()
                self.netD = nn.Sequential(nn.Linear(self.out_dim*2, self.out_dim),
                                                    nn.LeakyReLU(0.2,inplace=True),
                                                    nn.Linear(self.out_dim,1)
                                                    ).cuda()

                self.noise_ = torch.FloatTensor(self.config.batch, self.nz).cuda()
                self.noise_gen = torch.FloatTensor(self.config.batch, self.nz).cuda()

            elif self.config.unseen_prediction == 'mlp':
                self.netG = nn.Sequential(nn.Linear(self.out_dim,self.out_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(self.out_dim,self.out_dim)
                                                ).cuda()
            else:
                raise NotImplementedError(self.config.unseen_prediction)

        if self.config.contrastive_head:
            self.proj_head = nn.Sequential(nn.Linear(self.out_dim,self.out_dim),
                                           Normalize()).cuda()
        else:
            self.proj_head = nn.Identity().cuda()


        self._create_optimizer()
        self._init_dataloaders()

        self._init_memory_bank()

        if 'inat' in self.config.dataset:
            self._init_metrics()

        self._create_domain_centroids()
        if self.config.update_target_cls_weights:
            self._update_weight_seen_centroids_from_target()

        self.current_epoch = 0
        self.current_iteration = 0

    def dict_to_device(self, sample):
        sample_out = dict()
        for key, val in sample.items():
            if key != 'file':
                sample_out[key] = val.to(self.device)
        return sample_out
    def log(self,key, val):
        if self.logger is not None:
            self.logger.log({key:val})

    def train_one_epoch(self):

        
        num_batches = num_batches = max(len(self.source_train_set) // self.config.batch, len(self.target_train_set) // self.config.batch) + 1
        tqdm_batch = tqdm(
            total=num_batches, desc=f"[Epoch {self.current_epoch}]", leave=False, disable=None
        )
        tqdm_post = {}
        
        self.source_iter = iter(self.source_train_loader)
        self.target_iter = iter(self.target_train_loader)

        if self.config.use_source_as_cls_head:
            self.cls_head.fc.weight.data = self.source_centroids.as_tensor()
            for param in self.cls_head.parameters():
                param.requires_grad = False
        # for batch_i in range(5):
        for batch_i in range(num_batches):

            loss = 0
           
            sample_source = self.sample(domain='source')
            keypoints_source = sample_source['keypoints'] if self.config.is_pool_source_keypoints else None

            feat_source, _ = self.model(sample_source['x0'], keypoints_source)
            feat_source = F.normalize(feat_source, dim=1)

            sample_target = self.sample(domain='target')
            feat_target, _ = self.model(sample_target['x0'])
            feat_target = F.normalize(feat_target, dim=1)

            self.loss_fn.batch_size = sample_source['x0'].size(0)
            
            z_source = self.proj_head(feat_source)
            z_target = self.proj_head(feat_target)
            z_source_centroids =  self.proj_head(self.source_centroids.as_tensor())
            z_target_centroids =  self.proj_head(self.target_centroids.as_tensor())


            if self.config.proto_on_source > 0:
                loss_proto = self.loss_fn._compute_proto_loss(z_source, sample_source['label'], z_source_centroids)
                self.log('loss_proto/source',loss_proto.item())
                loss += self.config.proto_on_source * loss_proto

            if self.config.proto_on_target > 0:
                loss_proto = self.loss_fn._compute_proto_loss(z_target, sample_target['label'], z_target_centroids)
                self.log('loss_proto/target',loss_proto.item())
                loss += self.config.proto_on_target * loss_proto


            # compute instance losses only on seen classes target centroids
            index_source_seen = [x in self.seen_classes for x in sample_source['label'].cpu()]

            # self supervised loss cross domain
            if self.config.cross_on_source_instances > 0:
                loss_cross = self.loss_fn._compute_I2C_loss(z_source[index_source_seen],z_target_centroids)
                self.log('loss_cross/source',loss_cross.item())
                loss += self.config.cross_on_source_instances * loss_cross

            if self.config.cross_on_target_instances > 0:
                loss_cross = self.loss_fn._compute_I2C_loss(z_target,z_source_centroids)
                self.log('loss_cross/target',loss_cross.item())
                loss += self.config.cross_on_target_instances * loss_cross

            if self.config.proto_cross_on_source_instances > 0:
                loss_proto = self.loss_fn._compute_proto_loss(z_source[index_source_seen],
                                                            sample_source['label'][index_source_seen], z_target_centroids)
                self.log('loss_proto/source/cross',loss_proto.item())
                loss += self.config.cross_on_source_instances * loss_proto

            if self.config.proto_cross_on_target_instances > 0:
                loss_proto = self.loss_fn._compute_proto_loss(z_target, sample_target['label'], z_source_centroids)
                self.log('loss_proto/target/cross',loss_proto.item())
                loss += self.config.cross_on_target_instances * loss_proto


            # unseen cluster loss
            if self.config.unseen_centroid_on_target > 0:

                if self.config.unseen_prediction == 'mlp':
                    fake_centroid_target = self.netG(feat_source)
                    # loss_cluster = self.loss_fn._compute_cluster_loss(fake_centroid_target,sample_source['label'],self.target_centroids.as_tensor())
                    # self.log('loss_gen/mse_clusster',loss_cluster.item())

                    z_fake = self.proj_head(fake_centroid_target)

                    loss_cluster = self.loss_fn._compute_proto_loss(z_fake[index_source_seen],sample_source['label'][index_source_seen],self.target_centroids.as_tensor())
                    self.log('loss_proto/target_fake',loss_cluster.item())
                    loss += self.config.unseen_centroid_on_target * loss_cluster

                elif self.config.unseen_prediction == 'gan-gp':

                    # critic update
                    # TODO samples do not have the same class!
                    input_condition = feat_source #.detach()

                    criticD_real = self.netD(torch.cat([feat_target,input_condition],dim=-1))
                    criticD_real = criticD_real.mean()

                    self.noise_.normal_(0,1)
                    fake_centroid_target = self.netG(torch.cat([self.noise_,input_condition],dim=-1))
                    fake = fake_centroid_target
                    criticD_fake = self.netD(torch.cat([fake,input_condition],dim=-1))
                    criticD_fake = criticD_fake.mean()
                    
                    gradient_penalty = self.loss_fn._calc_gradient_penalty(self.netD,feat_target,fake.data,input_condition)
                    Wasserstein_D = criticD_real - criticD_fake
                    self.log('loss_gen/wasserstein_dist',Wasserstein_D.item())
                    D_cost = - Wasserstein_D + gradient_penalty # + cls_loss_real
                    loss += self.config.unseen_centroid_on_target * D_cost
            else:
                assert self.config.unseen_prediction == 'no'

            if self.config.superv_on_unseen_predicted_centroids > 0:
                assert self.config.unseen_centroid_on_target > 0
                y_pred_target_unseen = self.cls_head(fake_centroid_target[index_source_seen])
                loss_superv = self.criterion(y_pred_target_unseen,sample_source['label'][index_source_seen])
                self.log('loss_superv/fake_targets',loss_superv.item())
                loss += self.config.superv_on_unseen_predicted_centroids * loss_superv


            # supervised losses
            if self.config.superv_on_source > 0:
                y_pred_source = self.cls_head(feat_source)
                loss_superv = self.criterion(y_pred_source,sample_source['label'])
                self.log('loss_superv/source',loss_superv.item())
                loss += self.config.superv_on_source * loss_superv

            if self.config.superv_on_target > 0:
                y_pred_target = self.cls_head(feat_target)
                loss_superv = self.criterion(y_pred_target,sample_target['label'])
                self.log('loss_superv/target',loss_superv.item())
                loss += self.config.superv_on_target * loss_superv
            
            if self.config.superv_on_both_target_cls_head > 0:
                y_pred_target = F.linear(F.normalize(feat_target, dim=1),weight=self.target_centroids.as_tensor())
                loss_superv = self.criterion(y_pred_target,sample_target['label'])
                self.log('loss_superv/target_target_head',loss_superv.item())
                loss += self.config.superv_on_both_target_cls_head * loss_superv

                if np.any(index_source_seen):
                    y_pred_source = F.linear(F.normalize(feat_source[index_source_seen], dim=1),weight=self.target_centroids.as_tensor())
                    loss_superv = self.criterion(y_pred_source,sample_source['label'][index_source_seen])
                    self.log('loss_superv/source_target_head',loss_superv.item())
                    if  torch.isnan(loss_superv):
                        print(y_pred_source)
                        print(sample_source['label'][index_source_seen])
                        print(index_source_seen)
                        print('skipping loss_superv/source_target_head')
                    else:
                        loss += self.config.superv_on_both_target_cls_head * loss_superv

            assert not torch.isnan(loss)
            loss = loss / self.config.accum_grad_iters
            self.log('loss_train',loss.item())

            # Backpropagation
            loss.backward()
            if (batch_i % self.config.accum_grad_iters == 0) or (batch_i + 1 == num_batches):
                self.optim.step()
                self.optim.zero_grad()

            if  batch_i % 5 == 0 and self.config.unseen_prediction == 'gan-gp':
                # generator update
                input_condition = feat_source.detach()
                for p in self.netD.parameters():
                    p.requires_grad = False 

                self.netG.zero_grad()
                self.noise_gen.normal_(0, 1)
                fake_g_update = self.netG(torch.cat([self.noise_gen, input_condition],dim=-1))

                criticG_fake = self.netD(torch.cat([fake_g_update, input_condition], dim=-1))
                criticG_fake = criticG_fake.mean()
                G_cost = -criticG_fake

                loss_proto = self.loss_fn._compute_proto_loss(fake_g_update, sample_source['label'], self.target_centroids.as_tensor())
                self.log('loss_proto/fake_targets',loss_proto.item())

                lossG = G_cost + self.config.proto_on_source * loss_proto

                self.optimG.zero_grad()
                lossG.backward()
                self.optimG.step()

                for p in self.netD.parameters():
                    p.requires_grad = True


            # update memory bank

            self.source_bank.update(sample_source['index'],feat_source)
            self._update_domain_centroids('source',sample_source['label'])
                
            if self.config.skip_target_instance_bank:
                self._update_target_centroids_wo_instances(sample_target['label'],feat_target)
            else:
                self.target_bank.update(sample_target['index'],feat_target)
                self._update_domain_centroids('target',sample_target['label'])

            if self.config.update_target_cls_weights:
                self._update_weight_seen_centroids_from_target()

            if self.config.update_target_unseen_weights == 'iter':
                self._update_weight_unseen_centroids_from_source()

            if self.config.use_source_as_cls_head:
                self.cls_head.fc.weight.data = self.source_centroids.as_tensor()

            # adjust lr
            if self.optim_params['decay'] and self.config.optimizer == 'sgd':
                self.optim_iterdecayLR.step()


            tqdm_batch.set_postfix(tqdm_post)
            tqdm_batch.update()
            self.current_iteration += 1
            self.log('current_iteration',self.current_iteration)


            if self.current_iteration > self.config.max_iterations:
                tqdm_batch.close()
                print('max_iters reached')
                return None

        if self.config.update_target_unseen_weights == 'epoch':
            self._update_weight_unseen_centroids_from_source()

        tqdm_batch.close()
        self.current_epoch +=1

    def sample(self,domain):
        if domain =='source':
            try:
                sample = next(self.source_iter)
            except StopIteration:
                self.source_iter = iter(self.source_train_loader)
                sample = next(self.source_iter)
        elif domain =='target':
            try:
                sample = next(self.target_iter)
            except StopIteration:

                self.target_iter = iter(self.target_train_loader)
                sample = next(self.target_iter)

        sample_ = self.dict_to_device(sample)

        return sample_
    @torch.no_grad()
    def predict_features(self,loader):

        self.model.eval()
        data = []
        for sample in loader:
            sample = self.dict_to_device(sample)
            feat, _  = self.model(sample['x0'])
            feat = F.normalize(feat, dim=1)
            data.append((feat, sample['label'], sample['index']))

        return data

    def get_weights(self, cls_head):

        if cls_head == 'target_centroids_fake':
            weights = self.target_centroids.as_tensor().detach().clone()
            weights = self._update_weight_unseen_centroids_from_source(weights=weights)
        elif cls_head == 'seen_fromtarget_unseen_fromsource':
            weights = self.target_centroids.as_tensor().detach().clone()
            weights = self._update_weight_unseen_centroids_from_source(weights=weights, use_fake=False)
        elif cls_head == 'source_centroids':
            weights = self.source_centroids.as_tensor()
        elif cls_head == 'target_centroids':
            weights = self.target_centroids.as_tensor()
        elif cls_head == 'both_centroids':
            weights = torch.cat((self.source_centroids.as_tensor(),self.target_centroids.as_tensor()))
        elif cls_head == 'both_centroids_and_fake':
            syn_centroids = self._syn_centroids(self.source_centroids.as_tensor())
            weights = torch.cat((self.source_centroids.as_tensor(),self.target_centroids.as_tensor(),syn_centroids))
        elif cls_head == 'cls_head':
            self.cls_head.eval()
            weights = self.cls_head.fc.weight.data
        else:
            raise NotImplementedError
        return weights

    @torch.no_grad()
    def test(self, loader, cls_head = 'cls_head'):
        correct = 0
        n = 0
        self.model.eval()
        
        epoch_loss = AverageMeter()

        weights = self.get_weights(cls_head)

        for sample in loader:
            sample = self.dict_to_device(sample)
            labels = sample['label']

            feat, _  = self.model(sample['x0'])
            feat = F.normalize(feat, dim=1)
            output = F.linear(feat,weight=weights)

            loss = self.criterion(output, labels)
            pred = torch.max(output, dim=1)[1]
            if 'both' in cls_head:
                pred = torch.fmod(pred, self.num_class)

            correct += (pred == labels).sum().cpu().item()
            n += pred.size(0)

            epoch_loss.update(loss, pred.size(0))

        acc = correct / n
        return acc, epoch_loss.avg
    
    @torch.no_grad()
    def test_precomputed(self, data, cls_head = 'cls_head'):
        correct = 0
        n = 0
        
        epoch_loss = AverageMeter()

        weights = self.get_weights(cls_head)

        for feat, labels, index in data:

            output = F.linear(feat,weight=weights)

            loss = self.criterion(output, labels)
            pred = torch.max(output, dim=1)[1]
            if cls_head == 'both_centroids':
                pred = torch.fmod(pred, self.num_class)

            correct += (pred == labels).sum().cpu().item()
            n += pred.size(0)

            epoch_loss.update(loss, pred.size(0))

        acc = correct / n
        return acc, epoch_loss.avg

    def test_loop(self,epoch, cls_head='cls_head',suffix='',seen_tuple=None, unseen_tuple= None):

        if seen_tuple is None:
            acc_seen, loss_seen  = self.test(self.test_seen_loader,cls_head=cls_head)
        else:
            acc_seen, loss_seen = self.test_precomputed(seen_tuple, cls_head)
        if unseen_tuple is None:
            acc_unseen, loss_unseen = self.test(self.test_unseen_loader,cls_head=cls_head)
        else:
            acc_unseen, loss_unseen = self.test_precomputed(unseen_tuple, cls_head)
        if acc_seen+acc_unseen == 0:
            H = 0
            print('bug')
        else:
            H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)

        print(f'Epoch[{epoch}] {suffix} seen: {acc_seen:.4f} unseen: {acc_unseen:.4f} H: {H:.4f}')

        dict_log = {'val_epoch/acc_seen':acc_seen,
                    'val_epoch/acc_unseen': acc_unseen,
                    'val_epoch/H': H,
                    'val_epoch/loss_seen':loss_seen,
                    'val_epoch/loss_unseen':loss_unseen,
                    'epoch':epoch}

        dict_log_out = dict()
        for key, val in dict_log.items():
            if key != 'epoch':
                dict_log_out[key+suffix] = val
            else:
                dict_log_out[key] = val

        self.logger.log(dict_log_out)


    def _init_metrics(self):
        self.valid_topk = {'top1': torchmetrics.Accuracy(num_classes=self.num_class),
                            'top5':torchmetrics.Accuracy(num_classes=self.num_class, top_k=5,  multiclass=True),
                            'top10': torchmetrics.Accuracy(num_classes=self.num_class, top_k=10,  multiclass=True)}

        if self.index_per_hop is not None:
            for k in self.index_per_hop.keys():
                self.valid_topk = {**self.valid_topk,
                            k+'/top1': torchmetrics.Accuracy(num_classes=self.num_class),
                            k+'/top5':torchmetrics.Accuracy(num_classes=self.num_class, top_k=5,  multiclass=True),
                            k+'/top10': torchmetrics.Accuracy(num_classes=self.num_class, top_k=10,  multiclass=True)}
    def reset_metrics(self):
        for key in self.valid_topk.keys():
            self.valid_topk[key].reset()

    def add_batch_metrics(self, preds, target):
        for key in self.valid_topk.keys():
            if key.startswith('top'):
                self.valid_topk[key](preds, target)

    def add_batch_metrics_hops(self, preds, target,start, end=None):
        if end == None:
            index = start
        else:
            index = np.arange(start, end)
        for k, val in self.index_per_hop.items():
            index_bool = [x in val for x in index]
            if np.any(index_bool):
                for acc in ['top1','top5','top10']:
                    self.valid_topk[f'{k}/{acc}'](preds[index_bool], target[index_bool])

    def compute_metrics(self, prefix = '', valhops=False):
        output_ = dict()
        for key in self.valid_topk.keys():
            if key.startswith('top'):
                output_[prefix+key] = self.valid_topk[key].compute()
            elif valhops:
                output_[prefix+key] = self.valid_topk[key].compute()
        return output_

    @torch.no_grad()
    def test_precomputed_topk(self, data, cls_head = 'cls_head',prefix_m = '', valhops=False):
        self.reset_metrics()
        weights = self.get_weights(cls_head)

        for feat, labels, index in data:

            output = F.linear(feat,weight=weights)

            if cls_head == 'both_centroids':
                raise NotImplementedError
                # TODO check if it makes sense
                pred = pred.view(self.num_class,-1)
                pred = pred.max(dim=-1)
                
            self.add_batch_metrics(output.cpu(), labels.cpu())
            if self.index_per_hop is not None and valhops:
                self.add_batch_metrics_hops(output.cpu(), labels.cpu(), index.cpu())
        metrics = self.compute_metrics(prefix_m, valhops)
        return metrics


    def test_loop_topk(self,epoch, cls_head='cls_head',suffix='',seen_tuple=None, unseen_tuple= None):

        metrics_seen = self.test_precomputed_topk(seen_tuple, cls_head, prefix_m='seen_')
        metrics_unseen = self.test_precomputed_topk(unseen_tuple, cls_head, valhops=self.dataset_unseen.index_perhop, prefix_m='unseen_')
        metrics = {**metrics_seen,**metrics_unseen}

        acc_seen, acc_unseen = metrics['seen_top1'], metrics['unseen_top1']
        if acc_seen+acc_unseen == 0:
            H = 0
            print('bug')
        else:
            H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)

        print(f'Epoch[{epoch}] {suffix} seen: {acc_seen:.4f} unseen: {acc_unseen:.4f} H: {H:.4f}')
        metrics['H'] = H
        metrics['epoch'] = epoch

        p_ = 'val/'
        dict_log_out = dict()
        for key, val in metrics.items():
            if key != 'epoch':
                dict_log_out[p_+key+suffix] = val
            else:
                dict_log_out[p_+key] = val

        self.logger.log(dict_log_out)

    # compute train features
    @torch.no_grad()
    def compute_train_features(self, domain):

        self.model.eval()
        if domain == 'source':
            # train_set = self.source_train_set_noreplacement
            train_set = self.source_train_set
        else:
            train_set = self.target_train_set

        train_loader = DataLoader(train_set, batch_size=self.config.batch, shuffle=False, drop_last=False, num_workers=8)

        idx = torch.zeros(len(train_set), device=self.device, dtype=torch.int64)
        features = torch.zeros((len(train_set),self.config.latent_dim), device=self.device, dtype=torch.float32)
        y = torch.zeros(len(train_set), device=self.device, dtype=torch.int64)
        tqdm_batch = tqdm(
            total=len(train_loader), desc=f"[Compute train features of {domain}]"
        )

        for i, sample in enumerate(train_loader):
            start = i*self.config.batch
            sample = self.dict_to_device(sample)

            if domain == 'source' and self.config.is_pool_source_keypoints:
                atn_map = sample['keypoints'] # .to(self.device)
            else:
                atn_map = None
            feat, _ = self.model(sample['x0'], atn_map)
            feat = F.normalize(feat, dim=1)
            stop = start+feat.shape[0]
            features[start:stop] = feat
            y[start:stop] = sample['label']
            idx[start:stop] = sample['index']

            tqdm_batch.update()
        tqdm_batch.close()

        return idx, features, y



    # Memory bank
    def _init_dataloaders(self):

        # both domains
        self.train_set = TrainSet_billow(self.config, is_compute_mixlist=False)

        # dataset per domain
        self.source_train_set = self.train_set.dataset_billow

        self.target_train_set = self.train_set.dataset_target

        # resample source dset to have the same number of samples as target
        # self.source_train_set.resample_with_replacement(n=len(self.target_train_set))

        self.source_train_loader = DataLoader(self.source_train_set, batch_size=self.config.batch,
                                            sampler=RandomSampler(self.source_train_set, replacement=True, num_samples=len(self.target_train_set)),
                                            drop_last=True, num_workers=self.config.num_workers,
                                            pin_memory=True, persistent_workers=False)

        self.target_train_loader = DataLoader(self.target_train_set, batch_size=self.config.batch, shuffle=True,
                                                drop_last=True, num_workers=self.config.num_workers,
                                                pin_memory=True, persistent_workers=False)
        transform_cub = dataloaders_billow.data_transforms(self.config, transform_type='val')

        self.index_per_hop = None
        if not 'inat' in self.config.dataset:
            self.dataset_seen = Birds(split='seen', args=self.config, transform=transform_cub)
            self.dataset_unseen = Birds(split='unseen', args=self.config, transform=transform_cub)
        else:
            # self.dataset_seen = Webdataset_inat(opt=self.config, transform=transform_cub, split='val_seen')
            # self.dataset_unseen = Webdataset_inat(opt=self.config, transform=transform_cub, split='val_allhop')
            self.dataset_seen = H5Dataset_inat(args=self.config, transform=transform_cub, split='val_seen')
            self.dataset_unseen = H5Dataset_inat(args=self.config, transform=transform_cub, split='val_allhop')

            self.index_per_hop =  self.dataset_unseen.index_perhop
        # test_set = TestSet(domain_adaptation_task, repetition, sample_per_class)
        self.test_seen_loader = DataLoader(self.dataset_seen, batch_size=self.config.batch, shuffle=False, drop_last=False, num_workers=self.config.num_workers)
        self.test_unseen_loader = DataLoader(self.dataset_unseen, batch_size=self.config.batch, shuffle=False, drop_last=False, num_workers=self.config.num_workers)

        # print("Dataset Length Train : ", len(self.train_set)) #  " Test : ", len(test_set))

        self.seen_classes = [x for x in range(self.num_class) if x in self.train_set.y_target]
        self.unseen_classes = [x for x in range(self.num_class) if x not in self.seen_classes]


    @torch.no_grad()
    def _init_memory_bank(self):
        out_dim = self.out_dim
        
        # source domain 
        # data_len = len(self.source_train_set_noreplacement)
        data_len = len(self.source_train_set)
        memory_bank = MemoryBank(data_len, out_dim, self.config.momentum_source_instances)
        idx, feat, _ = self.compute_train_features(domain='source')
        memory_bank.update(idx, feat,m=0)
        self.source_bank = memory_bank

        if not self.config.skip_target_instance_bank:
            data_len = len(self.target_train_set)
            memory_bank = MemoryBank(data_len, out_dim, self.config.momentum_target_instances)
            idx, feat, _ = self.compute_train_features(domain='target')
            memory_bank.update(idx, feat, m = 0)
            self.target_bank = memory_bank


    # @torch.no_grad()
    # def _update_memory_bank(self, domain_name, indices, new_data_memory):
    #     memory_bank_wrapper = self.get_attr(domain_name, "memory_bank_wrapper")
    #     memory_bank_wrapper.update(indices, new_data_memory)
    #     updated_bank = memory_bank_wrapper.as_tensor()
    #     self.loss_fn.module.set_broadcast(domain_name, "memory_bank", updated_bank)

    # def _load_memory_bank(self, memory_bank_dict):
    #     """load memory bank from checkpoint

    #     Args:
    #         memory_bank_dict (dict): memory_bank dict of source and target domain
    #     """
    #     for domain_name in ("source", "target"):
    #         memory_bank = memory_bank_dict[domain_name]._bank.cuda()
    #         self.get_attr(domain_name, "memory_bank_wrapper")._bank = memory_bank
    #         self.loss_fn.module.set_broadcast(domain_name, "memory_bank", memory_bank)

    @torch.no_grad()
    def _create_domain_centroids(self):

        # source
        self.source_centroids = MemoryBank(self.num_class, self.out_dim, self.config.momentum_source_centroids)
        label = self.train_set.y_source
        feats = self.source_bank.as_tensor()
        centroids = torch.zeros((self.num_class,self.config.latent_dim), device=self.device, dtype=torch.float32)

        for c in range(self.num_class):
            mask_ = label == c
            centroid = F.normalize(torch.mean(feats[mask_], dim=0), dim=0)
            centroids[c] = centroid
        idx = torch.tensor(range(self.num_class)).cuda()

        self.source_centroids.update(idx, centroids, m = 0)


        self.target_centroids = MemoryBank(self.num_class, self.out_dim, self.config.momentum_target_centroids,
                                           init=self.config.init_type_target_centroids)
        label = self.train_set.y_target
        if self.config.skip_target_instance_bank:
            _, feats, _ = self.compute_train_features(domain='target')
        else:
            feats = self.target_bank.as_tensor()
        centroids = torch.zeros((len(self.seen_classes),self.config.latent_dim), device=self.device, dtype=torch.float32)

        for i, c in enumerate(self.seen_classes):
            mask_ = label == c
            centroid = F.normalize(torch.mean(feats[mask_], dim=0), dim=0)
            centroids[i] = centroid
        del feats
        idx = torch.tensor(self.seen_classes).cuda()

        self.target_centroids.update(idx, centroids, m = 0)


    @torch.no_grad()
    def _update_domain_centroids(self,domain,classes):
        # updates only classes updated in the batch

        classes = torch.unique(classes)

        labels = getattr(self.train_set,f'y_{domain}')
        
        centroidbank = self.get_attr(domain,'centroids')
        feats = self.get_attr(domain,'bank').as_tensor()

        is_forloop = False
        if is_forloop:
            centroids = torch.zeros((classes.shape[0],self.config.latent_dim), device=self.device, dtype=torch.float32)
            for i, c in enumerate(classes.cpu().numpy()):
                mask_ = labels == c
                centroid = F.normalize(torch.mean(feats[mask_], dim=0), dim=0)
                centroids[i] = centroid
        else:
            M = torch.zeros(self.num_class, len(feats), device = self.device)
            M[labels, torch.arange(len(feats))] = 1
            M = torch.nn.functional.normalize(M, p=1, dim=1)
            centroids = torch.mm(M, feats)[classes]
            centroids = F.normalize(centroids, dim=1)

        centroidbank.update(classes, centroids)

    @torch.no_grad()
    def _update_target_centroids_wo_instances(self,labels, feats):

        classes = torch.unique(labels)
        
        is_forloop = False
        if is_forloop:
            centroids = torch.zeros((classes.shape[0],self.config.latent_dim), device=self.device, dtype=torch.float32)
            
            for i, c in enumerate(classes.cpu().numpy()):
                mask_ = labels == c
                centroid = F.normalize(torch.mean(feats[mask_], dim=0), dim=0)
                centroids[i] = centroid
        else:
            M = torch.zeros(self.num_class, len(feats), device = self.device)
            M[labels, torch.arange(len(feats))] = 1
            M = torch.nn.functional.normalize(M, p=1, dim=1)
            centroids = torch.mm(M, feats)[classes]
            centroids = F.normalize(centroids, dim=1)

        self.target_centroids.update(classes,centroids)

    @torch.no_grad()
    def _update_weight_seen_centroids_from_target(self, weights = None):

        feats = self.target_bank.as_tensor()
        label = self.train_set.y_target

        if weights is None:
            weights = self.cls_head.fc.weight.data
        for c in self.seen_classes:
            mask_ = label == c
            weights[c] = F.normalize(torch.mean(feats[mask_], dim=0), dim=0)
        return weights

    @torch.no_grad()
    def _update_weight_unseen_centroids_from_source(self, weights = None, use_fake=True):

        centroids = self.source_centroids.as_tensor()
        if use_fake:
            centroids = self._syn_centroids(centroids)

        if weights is None:
            weights = self.cls_head.fc.weight.data
        for c in self.unseen_classes:
            # mask_ = label == c
            feats = centroids[c]
            weights[c] = F.normalize(feats, dim=0)
        return weights

    @torch.no_grad()
    def _syn_centroids(self, input_):

        if self.config.unseen_prediction == 'mlp':
            centroids = self.netG(input_)
        elif self.config.unseen_prediction == 'gan-gp':
            num = 100
            syn_noise = torch.FloatTensor(num, self.nz).cuda()
            centroids = torch.zeros(self.num_class,self.out_dim).cuda()
            for c in self.unseen_classes:
                syn_noise.normal_(0,1)
                input_condition = input_[c].repeat(num,1)
                centroid = self.netG(torch.cat([syn_noise, input_condition],dim=-1))
                centroids[c] = centroid.mean(dim=0)

        return centroids



    def _create_optimizer(self):

        optim_params = {
            # "learning_rate": 0.01,
            # "conv_lr_ratio": 0.1,
            "patience": 4,
            # "batch_size_lbd": 64,
            # "batch_size": 64,
            "decay": True,
            # "weight_decay": 5e-4,
            "cls_update": True,
            "nesterov":False,
            "momentum":True
          }

        self.optim_params = DotMap(optim_params)

        lr = self.config.learning_rate
        momentum = self.optim_params.momentum
        weight_decay = self.config.weight_decay
        # conv_lr_ratio = self.optim_params.conv_lr_ratio
        conv_lr_ratio = self.config.conv_lr_ratio

        parameters = []
        # batch_norm layer: no weight_decay
        params_bn, _ = torchutils.split_params_by_name(self.model, "bn")
        parameters.append({"params": params_bn, "weight_decay": 0.0})
        # conv layer: small lr
        _, params_conv = torchutils.split_params_by_name(self.model, ["fc", "bn"])
        if conv_lr_ratio:
            parameters[0]["lr"] = lr * conv_lr_ratio
            parameters.append({"params": params_conv, "lr": lr * conv_lr_ratio})
        else:
            parameters.append({"params": params_conv})
        # fc layer
        params_fc, _ = torchutils.split_params_by_name(self.model, "fc")
        if self.cls and self.optim_params.cls_update:
            params_fc.extend(list(self.cls_head.parameters()))
        if self.gen_unseen_centroid:
            params_fc.extend(list(self.netG.parameters()))
        
        params_fc.extend(list(self.proj_head.parameters()))
        
        parameters.append({"params": params_fc})


        if self.config.optimizer == 'sgd':
            self.optim = torch.optim.SGD(
                parameters,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                # nesterov=self.optim_params.nesterov,
            )
            # lr schedular
            if self.optim_params.lr_decay_schedule:
                optim_stepLR = torch.optim.lr_scheduler.MultiStepLR(
                    self.optim,
                    milestones=self.optim_params.lr_decay_schedule,
                    gamma=self.optim_params.lr_decay_rate,
                )
                self.lr_scheduler_list.append(optim_stepLR)

            if self.optim_params.decay:
                self.optim_iterdecayLR = torchutils.lr_scheduler_invLR(self.optim)

            if self.config.unseen_prediction == 'gan-gp':
                self.optimG = torch.optim.SGD(
                    self.netG.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    # nesterov=self.optim_params.nesterov,
                )

        elif self.config.optimizer == 'adam':
            self.optim = torch.optim.Adam(
                parameters,
                lr=lr,
                weight_decay=weight_decay)

            if self.config.unseen_prediction == 'gan-gp':
                self.optimG = torch.optim.Adam(
                    self.netG.parameters(),
                    lr=lr,
                    weight_decay=weight_decay)

        else:
            raise NotImplementedError(self.config.optimizer)


    def get_attr(self, domain, name):
        return getattr(self, f"{domain}_{name}")

    def set_attr(self, domain, name, value):
        setattr(self, f"{domain}_{name}", value)
        return self.get_attr(domain, name)

    def copy_checkpoint(self, filename="checkpoint.pth.tar"):
        if (
            self.copy_checkpoint_freq
            and self.current_epoch % self.copy_checkpoint_freq == 0
        ):
            # self.logger.info(f"Backup checkpoint_epoch_{self.current_epoch}.pth.tar")
            print(f"Backup checkpoint_epoch_{self.current_epoch}.pth.tar")
            torchutils.copy_checkpoint(
                filename=filename,
                folder=self.config.save_dir,
                copyname=f"checkpoint_epoch_{self.current_epoch}.pth.tar",
            )


    def load_checkpoint(
        self,
        filename,
        checkpoint_dir=None,
        load_memory_bank=False,
        load_model=True,
        load_optim=False,
        load_epoch=False,
        load_cls=True,
    ):
        checkpoint_dir = checkpoint_dir or self.config.save_dir
        filename = os.path.join(checkpoint_dir, filename)
        try:
            self.logger.info(f"Loading checkpoint '{filename}'")
            checkpoint = torch.load(filename, map_location="cpu")

            if load_epoch:
                self.current_epoch = checkpoint["epoch"]
                for domain_name in ("source", "target"):
                    self.set_attr(
                        domain_name,
                        "current_iteration",
                        checkpoint[f"iteration_{domain_name}"],
                    )
                self.current_iteration = checkpoint["iteration"]
                self.current_val_iteration = checkpoint["val_iteration"]

            if load_model:
                model_state_dict = checkpoint["model_state_dict"]
                self.model.load_state_dict(model_state_dict)

            if load_cls and self.cls and "cls_state_dict" in checkpoint:
                cls_state_dict = checkpoint["cls_state_dict"]
                self.cls_head.load_state_dict(cls_state_dict)

            if load_optim:
                optim_state_dict = checkpoint["optim_state_dict"]
                self.optim.load_state_dict(optim_state_dict)

                lr_pretrained = self.optim.param_groups[0]["lr"]
                lr_config = self.config.optim_params.learning_rate

                # Change learning rate
                if not lr_pretrained == lr_config:
                    for param_group in self.optim.param_groups:
                        param_group["lr"] = self.config.optim_params.learning_rate

            self._init_memory_bank()
            if (
                load_memory_bank or self.config.model_params.load_memory_bank == False
            ):  # load memory_bank
                self._load_memory_bank(
                    {
                        "source": checkpoint["memory_bank_source"],
                        "target": checkpoint["memory_bank_target"],
                    }
                )

            self.logger.info(
                f"Checkpoint loaded successfully from '{filename}' at (epoch {checkpoint['epoch']}) at (iteration s:{checkpoint['iteration_source']} t:{checkpoint['iteration_target']}) with loss = {checkpoint['loss']}\nval acc = {checkpoint['val_acc']}\n"
            )

        except OSError as e:
            self.logger.info(f"Checkpoint doesnt exists: [{filename}]")
            raise e

    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        
        target_bank = None if self.config.skip_target_instance_bank else self.target_bank

        out_dict = {
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optim.state_dict(),
            "source_bank": self.source_bank,
            "target_bank": target_bank,
            "source_centroids": self.source_centroids,
            "target_centroids": self.target_centroids,
            "epoch": self.current_epoch,
            "seen_classes": self.seen_classes,
            "unseen_classes":self.unseen_classes,
            # "iteration": self.current_iteration,
            # "iteration_source": self.get_attr("source", "current_iteration"),
            # "iteration_target": self.get_attr("target", "current_iteration"),
            # "val_iteration": self.current_val_iteration,
            # "val_acc": np.array(self.val_acc),
            # "val_metric": self.current_val_metric,
            # "loss": self.current_loss,
            # "train_loss": np.array(self.train_loss),
        }
        if self.cls:
            out_dict["cls_state_dict"] = self.cls_head.state_dict()
        # best according to source-to-target
        is_best = False 
        # (
        #     self.current_val_metric == self.best_val_metric
        # ) or not self.config.validate_freq
        torchutils.save_checkpoint(
            out_dict, is_best, filename=filename, folder=self.config.save_dir
        )
        self.copy_checkpoint()