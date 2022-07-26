from __future__ import print_function
import argparse
import sys
sys.path.append("..")
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import util
import dataloader_inat
import classifier_embed_contras
import model
import losses
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import tqdm

import pprint
pp = pprint.PrettyPrinter(width=41, compact=True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='FLO')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')

parser.add_argument('--class_embedding', default='sent',help='att or sent')
parser.add_argument('--code_path', default=None, help='path of codes from billow')
parser.add_argument('--billow_images', action='store_true', default=False, help='use images directly')
parser.add_argument('--datasplit', default='zsl17', choices=['zsl17','dna'])
parser.add_argument('--normalize_embedding',default='no', choices=['no','mean_var','exp','max','l2'], help='')

parser.add_argument('--notes', default=None, help='notes for special settings')

parser.add_argument('--valsplit', default='1hop', choices=['1hop','2hop','3hop','4hop','allhop'], help='only for inaturalist datasets')
parser.add_argument('--filter_using_comm_names', action='store_true', default=False, help='use images files to obtain the classes directly, only for CUB_billow and CUB_dna_billow')

parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', type=bool, default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', type=bool, default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=2048, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024 , help='size of semantic features')
parser.add_argument('--nz', type=int, default=1024, help='noise for generation')
parser.add_argument('--embedSize', type=int, default=2048, help='size of embedding h')
parser.add_argument('--outzSize', type=int, default=512, help='size of non-liner projection z')

## network architechure
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator G')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator D')
parser.add_argument('--nhF', type=int, default=2048, help='size of the hidden units comparator network F')

parser.add_argument('--ins_weight', type=float, default=0.001, help='weight of the classification loss when learning G')
parser.add_argument('--cls_weight', type=float, default=0.001, help='weight of the score function when learning G')
parser.add_argument('--ins_temp', type=float, default=0.1, help='temperature in instance-level supervision')
parser.add_argument('--cls_temp', type=float, default=0.1, help='temperature in class-level supervision')

parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to training')
parser.add_argument('--lr_decay_epoch', type=int, default=100, help='conduct learning rate decay after every 100 epochs')
parser.add_argument('--lr_dec_rate', type=float, default=0.99, help='learning rate decay rate')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=3483, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--nclass_seen', type=int, default=150, help='number of all classes')

parser.add_argument('--gpus', default='0', help='the number of the GPU to use')
opt = parser.parse_args()
pp.pprint(opt.__dict__)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus

if opt.manualSeed or opt.manualSeed == -1:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
if opt.dataset in ['inat2017','inat2021','inat2021mini']:
    train_data = dataloader_inat.H5Dataset_inat(split='train', opt=opt, is_load_in_memory=True)
    val_seen_data = dataloader_inat.H5Dataset_inat(split='val_seen', opt=opt,is_load_in_memory=True)
    val_unseen_data = dataloader_inat.H5Dataset_inat(split='val_'+opt.valsplit, opt=opt,is_load_in_memory=True)

    data = util.Data_empty(ntrain=len(train_data),
                            seenclasses = train_data.seen_classes_id,
                            unseenclasses = val_unseen_data.classes_split_id,
                            attribute_seen= train_data.billow_codes.data,
                            attribute_unseen = val_unseen_data.billow_codes.data,
                            train_feature = train_data.data_feat,
                            train_label = train_data.data_label,
                            test_seen_feature = val_seen_data.data_feat,
                            test_seen_label = val_seen_data.data_label,
                            test_unseen_feature = val_unseen_data.data_feat,
                            test_unseen_label = val_unseen_data.data_label,
                            test_unseen_indexperhop = val_unseen_data.index_perhop
                            )

else:
    data = util.DATA_LOADER(opt)

    train_data = util.MyDataset(X = data.train_feature, Y = data.train_label, att = data.attribute, opt=opt)
    
    print("# of training samples: ", data.ntrain)

train_loader = DataLoader(train_data, batch_size=opt.batch_size,
                                            sampler=RandomSampler(train_data, replacement=True, num_samples=int(1e100)),
                                            drop_last=True, num_workers=8,
                                            pin_memory=True, persistent_workers=False)
train_dset_iter = iter(train_loader)



netG = model.MLP_G(opt)
netMap = model.Embedding_Net(opt)
netD = model.MLP_CRITIC(opt)
F_ha = model.Dis_Embed_Att(opt)
if opt.billow_images:
    netG.backbone, input_size = model.initialize_model(model_name=opt.code_path)


save_directory_path = os.path.join(opt.outf,'CE_GZSL','version')
save_directory_path = util.add_version_path(save_directory_path)

is_wandb = True
if is_wandb:
    import wandb
    wandb.init(config=opt,project="LrsGAN",
                entity="birds_ethz",
                dir=save_directory_path,
                settings=wandb.Settings(start_method="fork")
                )
    wandb.log({"model":"CE_GZSL"})

    writer = wandb
    def add_scalar(key, val, global_step):
        writer.log({key:val,'global_step':global_step})
    writer.add_scalar = add_scalar

else:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=save_directory_path)



if len(opt.gpus.split(','))>1:
    netG=nn.DataParallel(netG)
    netD = nn.DataParallel(netD)
    netMap = nn.DataParallel(netMap)
    F_ha = nn.DataParallel(F_ha)


contras_criterion = losses.SupConLoss_clear(opt.ins_temp)

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise_gen = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)

if opt.billow_images:
    input_image = torch.FloatTensor(opt.batch_size, 3, 256, 256)

if opt.cuda:
    netG.cuda()
    netD.cuda()
    netMap.cuda()
    F_ha.cuda()
    input_res = input_res.cuda()
    noise_gen, input_att = noise_gen.cuda(), input_att.cuda()
    input_label = input_label.cuda()
    if opt.billow_images:
        input_image = input_image.cuda()


def sample():
    # batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    batch_feature, batch_label, batch_att = next(train_dset_iter)
    input_res.copy_(batch_feature)
    if opt.billow_images:
        input_image.copy_(batch_att)
    else:
        input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        if opt.dataset.startswith('inat'):
            iclass_att = attribute[i]
        else:
            iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG(syn_noise, syn_att)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label

@torch.no_grad()
def generate_syn_feature_billow(netG, classes, attribute, num):
    nclass = classes.size(0)
    # syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    # syn_label = torch.LongTensor(nclass * num)
    syn_feature = []
    syn_label = []
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_image = train_data.data_billow.get_samples_class(iclass.numpy(), is_random=False)
        # iclass_att = attribute[iclass]
        for image in iclass_image:
            iclass_att = netG.backbone(image.cuda().unsqueeze(0)).squeeze()
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, 1)
            output = netG(syn_noise, syn_att)
            syn_feature.append(output.data.cpu())
            syn_label.append(iclass.repeat(num))         
            # syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
            # syn_label.narrow(0, i * num, num).fill_(iclass)
    syn_feature = torch.cat(syn_feature)
    syn_label = torch.cat(syn_label)
    return syn_feature, syn_label

# setup optimizer
import itertools

optimizerD = optim.Adam(itertools.chain(netD.parameters(), netMap.parameters(), F_ha.parameters()), lr=opt.lr,
                        betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    # print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, input_att)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

# use the for-loop to save the GPU-memory
def class_scores_for_loop(embed, input_label, relation_net):
    all_scores=torch.FloatTensor(embed.shape[0],opt.nclass_seen).cuda()
    for i, i_embed in enumerate(embed):
        expand_embed = i_embed.repeat(opt.nclass_seen, 1)#.reshape(embed.shape[0] * opt.nclass_seen, -1)
        all_scores[i]=(torch.div(relation_net(torch.cat((expand_embed, data.attribute_seen.cuda()), dim=1)),opt.cls_temp).squeeze())
    score_max, _ = torch.max(all_scores, dim=1, keepdim=True)
    # normalize the scores for stable training
    scores_norm = all_scores - score_max.detach()
    mask = F.one_hot(input_label, num_classes=opt.nclass_seen).float().cuda()
    exp_scores = torch.exp(scores_norm)
    log_scores = scores_norm - torch.log(exp_scores.sum(1, keepdim=True))
    cls_loss = -((mask * log_scores).sum(1) / mask.sum(1)).mean()
    return cls_loss

# It is much faster to use the matrix, but it cost much GPU memory.
def class_scores_in_matrix(embed, input_label, relation_net):
    expand_embed = embed.unsqueeze(dim=1).repeat(1, opt.nclass_seen, 1).reshape(embed.shape[0] * opt.nclass_seen, -1)
    expand_att = data.attribute_seen.unsqueeze(dim=0).repeat(embed.shape[0], 1, 1).reshape(
        embed.shape[0] * opt.nclass_seen, -1).cuda()
    all_scores = torch.div(relation_net(torch.cat((expand_embed, expand_att), dim=1)),opt.cls_temp).reshape(embed.shape[0],
                                                                                                    opt.nclass_seen)
    score_max, _ = torch.max(all_scores, dim=1, keepdim=True)
    scores_norm = all_scores - score_max.detach()
    mask = F.one_hot(input_label, num_classes=opt.nclass_seen).float().cuda()
    exp_scores = torch.exp(scores_norm)
    log_scores = scores_norm - torch.log(exp_scores.sum(1, keepdim=True))
    cls_loss = -((mask * log_scores).sum(1) / mask.sum(1)).mean()
    return cls_loss


for epoch in tqdm.trange(opt.nepoch):
    FP = 0
    mean_lossD = 0
    mean_lossG = 0
    # data.ntrain = 200
    for i in tqdm.trange(0, data.ntrain, opt.batch_size):
        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in netMap.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in F_ha.parameters():  # reset requires_grad
            p.requires_grad = True

        for iter_d in range(opt.critic_iter):
            sample()
            netD.zero_grad()
            netMap.zero_grad()
            #
            # train with realG
            # sample a mini-batch
            # sparse_real = opt.resSize - input_res[1].gt(0).sum()
            if opt.billow_images:
                input_att = netG.backbone(input_image).squeeze()
                input_att = F.normalize(input_att, dim=1)
            embed_real, outz_real = netMap(input_res)
            criticD_real = netD(input_res, input_att)
            criticD_real = criticD_real.mean()

            # CONTRASITVE LOSS
            real_ins_contras_loss = contras_criterion(outz_real, input_label)

            # train with fakeG
            noise_gen.normal_(0, 1)
            fake = netG(noise_gen, input_att)
            fake_norm = fake.data[0].norm()
            sparse_fake = fake.data[0].eq(0).sum()
            criticD_fake = netD(fake.detach(), input_att)
            criticD_fake = criticD_fake.mean()

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            Wasserstein_D = criticD_real - criticD_fake

            if opt.billow_images:
                cls_loss_real = 0
                ## TODO implement the class scores
            else:
                cls_loss_real = class_scores_for_loop(embed_real, input_label, F_ha)
    
            D_cost = criticD_fake - criticD_real + gradient_penalty + real_ins_contras_loss + cls_loss_real

            D_cost.backward()
            optimizerD.step()
        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False  # avoid computation
        for p in netMap.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in F_ha.parameters():  # reset requires_grad
            p.requires_grad = False

        netG.zero_grad()
        noise_gen.normal_(0, 1)
        if opt.billow_images:
            input_att = netG.backbone(input_image).squeeze()
            input_att = F.normalize(input_att, dim=1)
        fake = netG(noise_gen, input_att)

        embed_fake, outz_fake = netMap(fake)

        criticG_fake = netD(fake, input_att)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake

        embed_real, outz_real = netMap(input_res)

        all_outz = torch.cat((outz_fake, outz_real.detach()), dim=0)

        fake_ins_contras_loss = contras_criterion(all_outz, torch.cat((input_label, input_label), dim=0))

        if opt.billow_images:
            cls_loss_fake = 0
        else:
            cls_loss_fake = class_scores_for_loop(embed_fake, input_label, F_ha)

        errG = G_cost + opt.ins_weight * fake_ins_contras_loss + opt.cls_weight * cls_loss_fake  # + opt.ins_weight * c_errG

        errG.backward()
        optimizerG.step()

    F_ha.zero_grad()
    if (epoch + 1) % opt.lr_decay_epoch == 0:
        for param_group in optimizerD.param_groups:
            param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
        for param_group in optimizerG.param_groups:
            param_group['lr'] = param_group['lr'] * opt.lr_dec_rate

    mean_lossG /= data.ntrain / opt.batch_size
    mean_lossD /= data.ntrain / opt.batch_size
    print(
        '[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, real_ins_contras_loss:%.4f, fake_ins_contras_loss:%.4f, cls_loss_real: %.4f, cls_loss_fake: %.4f'
        % (epoch, opt.nepoch, D_cost, G_cost, Wasserstein_D, real_ins_contras_loss, fake_ins_contras_loss, cls_loss_real, cls_loss_fake))
    writer.add_scalar('Loss_D',D_cost.item(), epoch)    
    writer.add_scalar('Loss_G',G_cost.item(), epoch)    
    writer.add_scalar('Wasserstein_dist',Wasserstein_D.item(), global_step=epoch)    
    # writer.add_scalar('c_errG',c_errG.item(), global_step=epoch)    
    # writer.add_scalar('E_dist',Euclidean_loss.item(), global_step=epoch)    
    # writer.add_scalar('Corr_Loss',Correlation_loss.item(), global_step=epoch)    

    # evaluate the model, set G to evaluation mode
    netG.eval()

    for p in netMap.parameters():  # reset requires_grad
        p.requires_grad = False

    if opt.gzsl: # Generalized zero-shot learning
        if opt.dataset.startswith('inat'):
            syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute_unseen, opt.syn_num)
            train_X = torch.cat((data.train_feature, syn_feature), 0)
            train_Y = torch.cat((data.train_label, syn_label), 0)

            nclass = opt.nclass_all
            cls = classifier_embed_contras.CLASSIFIER(train_X, train_Y, netMap, opt.embedSize, data, nclass, opt.cuda,
                                                    opt.classifier_lr, 0.5, 25, opt.syn_num,
                                                    True, topk=True, index_per_hop=data.test_unseen_indexperhop)
        
            print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.m['seen_top1'], cls.m['seen_top1'], cls.m['H']))
            for key, val in cls.m.items():
                writer.add_scalar('test/'+key,val, global_step=epoch)
            for key, val in cls.best_m.items():
                writer.add_scalar('test/best/'+key,val, global_step=epoch)

        else:
            if opt.billow_images:
                syn_feature, syn_label = generate_syn_feature_billow(netG, data.unseenclasses, data.attribute, opt.syn_num)
            else:
                syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
            train_X = torch.cat((data.train_feature, syn_feature), 0)
            train_Y = torch.cat((data.train_label, syn_label), 0)

            nclass = opt.nclass_all
            cls = classifier_embed_contras.CLASSIFIER(train_X, train_Y, netMap, opt.embedSize, data, nclass, opt.cuda,
                                                    opt.classifier_lr, 0.5, 25, opt.syn_num,
                                                    True)
            print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
            writer.add_scalar('test/unseen_acc',cls.acc_unseen, global_step=epoch)    
            writer.add_scalar('test/seen_acc',cls.acc_seen, global_step=epoch)    
            writer.add_scalar('test/H',cls.H, global_step=epoch)    
            
            writer.add_scalar('test/best/unseen_acc',cls.best_acc_unseen, global_step=epoch)    
            writer.add_scalar('test/best/seen_acc',cls.best_acc_seen, global_step=epoch)    
            writer.add_scalar('test/best/H',cls.best_H, global_step=epoch)    
            

    else:  # conventional zero-shot learning
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        cls = classifier_embed_contras.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), netMap,
                                                  opt.embedSize, data,
                                                  data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 100,
                                                  opt.syn_num,
                                                  False)
        acc = cls.acc
        print('unseen class accuracy=%.4f '%acc)


    # reset G to training mode
    netG.train()
    for p in netMap.parameters():  # reset requires_grad
        p.requires_grad = True

