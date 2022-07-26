# from pkgutil import get_data
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
import os
import sys
import wandb
# from dotmap import DotMap

import random
import torch.backends.cudnn as cudnn

# from dataloader import TrainSet, TestSet, TrainSet_billow
import dataloaders_billow
import utils
# from models.memorybank import MemoryBank
from experiment import ExperimentMemoryBank
import pprint
pp = pprint.PrettyPrinter(width=41, compact=True)

# domain_adaptation_task = 'MNIST_to_USPS'
# sample_per_class = 7
# repetition = 0
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset", default="billow_cub", type=str)
parser.add_argument("--img_size", default=256, type=int)
parser.add_argument("--std_norm", default=0.5, type=int)
parser.add_argument('--billow_norm', default='basic',choices=['basic','basic_trainmean'])
parser.add_argument("--backbone", default="resnet18", type=str)
parser.add_argument('--is_pool_source_keypoints', default=False,action='store_true')

parser.add_argument('--filter_using_comm_names', action='store_true', default=False, help='use images files to obtain the classes directly, only for CUB_billow and CUB_dna_billow')
parser.add_argument("--checkpoints_dir", default=dataloaders_billow.get_basepath()+'birds/DA_baseline/logs', type=str)
parser.add_argument("--model", default="DA_memorybank", type=str)

parser.add_argument('--run_id', type = str, default=None)
parser.add_argument('--notes', type = str, default=None)
parser.add_argument("--load_pretrain_dir", default=None, type=str)
# parser.add_argument('--ce_loss_ontarget', default=True)
parser.add_argument("--batch", default=128, type=int)
parser.add_argument("--max_iterations", default=200000,type=int)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--accum_grad_iters", type=int, default=1)

parser.add_argument('--optimizer', default='sgd',choices=['sgd','adam'])
parser.add_argument("--learning_rate", default=0.01, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--conv_lr_ratio", default=0.01, type=float)

parser.add_argument("--latent_dim", default=512, type=int)

parser.add_argument('--proto_on_source', default=0.0, type=float)
parser.add_argument('--proto_on_target', default=0.0, type=float)


parser.add_argument('--cross_on_source_instances', default=0.0, type=float)
parser.add_argument('--cross_on_target_instances', default=0.0, type=float)

parser.add_argument('--proto_cross_on_target_instances', default=0.0, type=float)
parser.add_argument('--proto_cross_on_source_instances', default=0.0, type=float)


parser.add_argument('--freeze_backbone', default=False,action='store_true')

parser.add_argument('--billow_from_numpy', default=False,action='store_true')
parser.add_argument('--load_inat_in_memory', default=False,action='store_true')

parser.add_argument('--momentum_source_instances', default=0.0, type=float, help='momentum to update the memory bank on source instances')
parser.add_argument('--momentum_target_instances', default=0.0, type=float, help='momentum to update the memory bank on target instances')
parser.add_argument('--momentum_source_centroids', default=0.0, type=float)
parser.add_argument('--momentum_target_centroids', default=0.0, type=float)

parser.add_argument('--skip_target_instance_bank', default=False,action='store_true')
parser.add_argument('--init_type_target_centroids', default="random",choices=['random','zeros'])
parser.add_argument('--superv_on_source', default=1.0, type=float)
parser.add_argument('--superv_on_target', default=0.0, type=float)

parser.add_argument('--superv_on_both_target_cls_head', default=0.0, type=float)
parser.add_argument('--unseen_centroid_on_target', default=0.0, type=float)
parser.add_argument('--superv_on_unseen_predicted_centroids', default=0.0, type=float)
parser.add_argument('--unseen_prediction', default='no',choices=['no','mlp','gan-gp'])

parser.add_argument('--use_source_as_cls_head', default=False,action='store_true')

parser.add_argument('--update_target_unseen_weights', default='no',choices=['no','iter','epoch'])

parser.add_argument('--update_target_cls_weights', default=False,action='store_true')

parser.add_argument('--contrastive_head', default=False,action='store_true')

parser.add_argument("--manualSeed", default=None, type=int)
parser.add_argument('--copy_checkpoint_freq', default=None, type=int)
params = parser.parse_args()


save_dir = utils.add_version_path(out_dir = os.path.join(params.checkpoints_dir, 'version_'),timestamp=True)

params.save_dir = save_dir

params.LSB_JOBID = os.environ.get('LSB_JOBID')
params.LSB_JOBINDEX = os.environ.get('LSB_JOBINDEX')

if params.num_workers == -1:
    params.num_workers = os.cpu_count()
    print('updated num_workers to', params.num_workers)
pp.pprint(params.__dict__)
project = "DA_mbank_inat" if 'inat' in params.dataset else "DA_mbank"

if params.manualSeed == -1:
    params.manualSeed = random.randint(1, 10000)
if params.manualSeed is not None:
    print("Random Seed: ", params.manualSeed)
    random.seed(params.manualSeed)
    torch.manual_seed(params.manualSeed)
    torch.cuda.manual_seed_all(params.manualSeed)

cudnn.benchmark = True

wandb.init(config=params,
            settings=wandb.Settings(start_method="fork"),
            project=project,
            entity="birds_ethz",
            dir=save_dir)
logger = wandb

experiment = ExperimentMemoryBank(params, logger=logger)

for epoch in range(params.epochs):
    print("Epoch %d"%(epoch))

    experiment.train_one_epoch()

    seenfeats = experiment.predict_features(experiment.test_seen_loader)
    unseenfeats = experiment.predict_features(experiment.test_unseen_loader)

    if 'CUB' in params.dataset or 'cub' in params.dataset:
        experiment.test_loop(epoch, seen_tuple=seenfeats, unseen_tuple=unseenfeats)

        if epoch % 2 == 0:
            experiment.test_loop(epoch, cls_head='source_centroids',suffix='_source', seen_tuple=seenfeats, unseen_tuple=unseenfeats)
            # experiment.test_loop(epoch, cls_head='target_centroids',suffix='_target')
            experiment.test_loop(epoch, cls_head='seen_fromtarget_unseen_fromsource',suffix='_starget_usource', seen_tuple=seenfeats, unseen_tuple=unseenfeats)
            experiment.test_loop(epoch, cls_head='both_centroids',suffix='_both', seen_tuple=seenfeats, unseen_tuple=unseenfeats)

            if experiment.gen_unseen_centroid:
                experiment.test_loop(epoch, cls_head='both_centroids_and_fake',suffix='_both_and_fake', seen_tuple=seenfeats, unseen_tuple=unseenfeats)
                experiment.test_loop(epoch, cls_head='target_centroids_fake',suffix='_target_fake', seen_tuple=seenfeats, unseen_tuple=unseenfeats)
    else:
        experiment.test_loop_topk(epoch, seen_tuple=seenfeats, unseen_tuple=unseenfeats)
    # experiment.save_checkpoint()
    if experiment.current_iteration > params.max_iterations:
        break

experiment.save_checkpoint(filename='checkpoint_last.pth.tar')