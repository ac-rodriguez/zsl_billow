import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
from sklearn.preprocessing import MinMaxScaler 
import sys
import torchmetrics
import tqdm
class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, map_net, embed_size, data_loader, _nclass,
                _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True,
                topk =False, index_per_hop=None):
        self.train_X =  _train_X 
        self.train_Y = _train_Y 
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label 
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label 
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.MapNet=map_net
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = embed_size
        self.cuda = _cuda
        self.topk = topk
        self.index_per_hop = index_per_hop
        self.model =  LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.model.apply(util.weights_init)
        self.criterion = nn.NLLLoss()
        
        self.input = torch.FloatTensor(_batch_size, _train_X.size(1))
        self.label = torch.LongTensor(_batch_size) 
        
        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        if self.topk:
            self.valid_topk = {'top1': torchmetrics.Accuracy(num_classes=self.nclass),
                                'top5':torchmetrics.Accuracy(num_classes=self.nclass, top_k=5,  multiclass=True),
                                'top10': torchmetrics.Accuracy(num_classes=self.nclass, top_k=10,  multiclass=True)}

            if self.index_per_hop is not None:
                for k in self.index_per_hop.keys():
                    self.valid_topk = {**self.valid_topk,
                                k+'/top1': torchmetrics.Accuracy(num_classes=self.nclass),
                                k+'/top5':torchmetrics.Accuracy(num_classes=self.nclass, top_k=5,  multiclass=True),
                                k+'/top10': torchmetrics.Accuracy(num_classes=self.nclass, top_k=10,  multiclass=True)}

            if generalized:
                self.best_m, self.m = self.fit_topk()
        else:
            if generalized:
                results = self.fit()
                self.best_acc_seen, self.best_acc_unseen, self.best_H = results[0]
                self.acc_seen, self.acc_unseen, self.H = results[1]
            else:
                self.acc = self.fit_zsl()
    
    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                   
                embed, _=self.MapNet(self.input)
                output = self.model(embed)
                loss = self.criterion(output, self.label)
                mean_loss += loss.data
                loss.backward()
                self.optimizer.step()
            acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            if acc > best_acc:
                best_acc = acc
        print('Training classifier loss= %.4f' % (loss))
        return best_acc 

    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                embed, _ = self.MapNet(self.input)
                output = self.model(embed)
                loss = self.criterion(output, self.label)

                loss.backward()
                self.optimizer.step()
            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            if (acc_seen+acc_unseen)==0:
                print('a bug')
                H=0
            else:
                H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
        return (best_seen, best_unseen, best_H), (acc_seen, acc_unseen, H)
    def fit_topk(self):
        best_H = 0
        best_metrics = None

        # best_seen = 0
        # best_unseen = 0
        for epoch in tqdm.trange(self.nepoch,desc='Evaluating'):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                embed, _ = self.MapNet(self.input)
                output = self.model(embed)
                loss = self.criterion(output, self.label)

                loss.backward()
                self.optimizer.step()
            metrics_seen = self.val_gzsl_topk(self.test_seen_feature, self.test_seen_label, self.seenclasses, prefix_m='seen_')
            metrics_unseen = self.val_gzsl_topk(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses,prefix_m='unseen_',valhops=True)
            metrics = {**metrics_seen,**metrics_unseen}
            acc_seen, acc_unseen = metrics['seen_top1'], metrics['unseen_top1']
            if (acc_seen+acc_unseen)==0:
                print('a bug')
                H=0
            else:
                H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
                metrics['H'] = H
            if H > best_H or best_metrics is None:
                best_metrics = metrics
                best_H = H
        return best_metrics, metrics
        # return 
        # return (best_seen, best_unseen, best_H), (acc_seen, acc_unseen, H)                  
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]


    def val_gzsl(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    embed, _ = self.MapNet(test_X[start:end].cuda())
                    output = self.model(embed)
                else:
                    embed, _ = self.MapNet(test_X[start:end])
                    output = self.model(embed)
            _, predicted_label[start:end] = torch.max(output, 1)
            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
        acc_per_class /= target_classes.size(0)
        return acc_per_class 

    def reset_metrics(self):
        for key in self.valid_topk.keys():
            self.valid_topk[key].reset()

    def add_batch_metrics(self, preds, target):
        for key in self.valid_topk.keys():
            if key.startswith('top'):
                self.valid_topk[key](preds, target)

    def add_batch_metrics_hops(self, preds, target,start, end):
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
        # acc = self.valid_acc.compute()
        # acc_5 = self.valid_acc_5.compute()
        # acc_10 = self.valid_acc_10.compute()
        
        # return {prefix+'acc':acc,prefix+'acc_5':acc_5,prefix+'acc_10':acc_10}

    def val_gzsl_topk(self, test_X, test_label, target_classes, prefix_m = '', valhops=False): 
        start = 0
        ntest = test_X.size()[0]
        self.reset_metrics()

        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    embed, _ = self.MapNet(test_X[start:end].cuda())
                    output = self.model(embed)
                else:
                    embed, _ = self.MapNet(test_X[start:end])
                    output = self.model(embed)
            self.add_batch_metrics(output.cpu(), test_label[start:end])
            if self.index_per_hop is not None and valhops:
                self.add_batch_metrics_hops(output.cpu(), test_label[start:end],start,end)
            start = end
        metrics = self.compute_metrics(prefix_m, valhops)
        return metrics
    # test_label is integer 
    def val(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    embed, _ = self.MapNet(test_X[start:end].cuda())
                    output = self.model(embed)
                else:
                    embed, _ = self.MapNet(test_X[start:end])
                    output = self.model(embed)
            _, predicted_label[start:end] = torch.max(output, 1)
            start = end

        acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = float(torch.sum(test_label[idx]==predicted_label[idx])) / float(torch.sum(idx))
        return acc_per_class.mean() 

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o  
