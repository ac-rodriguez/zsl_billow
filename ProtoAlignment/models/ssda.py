import random

import numpy as np
import torch
import torch.cuda.comm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# from pcs.utils import reverse_domain, torchutils
import torchutils
# import reverse_domain
import torch.autograd as autograd

class SSDALossModule(torch.nn.Module):
    def __init__(self, config, gpu_devices=[0]):
        super(SSDALossModule, self).__init__()
        self.config = config
        self.gpu_devices = gpu_devices

        # self.k = config.loss_params.k
        # self.t = config.loss_params.temp
        self.m = 0.1
        # self.loss_w = self.config.loss_params.weight
        # self.loss = self.config.loss_params.loss

        self.indices = None
        self.outputs = None
        self.indices_random = None
        self.batch_size = None
        self.mse_loss = nn.MSELoss()

        self.gp_loss = GradientPaneltyLoss()

        self.broadcast_name = []

    def get_attr(self, domain, name):
        return getattr(self, name + "_" + domain)

    def set_attr(self, domain, name, value):
        setattr(self, name + "_" + domain, value)
        return self.get_attr(domain, name)

    def set_broadcast(self, domain_name, name, value, comm=False):
        if isinstance(value, list):
            value = value.copy()
            for i in range(len(value)):
                value[i] = torch.cuda.comm.broadcast(value[i], self.gpu_devices)
        else:
            value = torch.cuda.comm.broadcast(value, self.gpu_devices)

        self.set_attr(domain_name, f"{name}_broadcast", value)
        self.set_attr(domain_name, name, None)
        self.broadcast_name.append((domain_name, name))

    def get_broadcast(self, domain_name, name, gpu_idx):
        broadcast = self.get_attr(domain_name, f"{name}_broadcast")
        if len(broadcast) == 0 or isinstance(broadcast[0], list):
            tmp = []
            for x in broadcast:
                tmp.append(x[gpu_idx])
            self.set_attr(domain_name, name, tmp)
        else:
            tmp = broadcast[gpu_idx]
            self.set_attr(domain_name, name, tmp)
        return tmp

    def _identify_gpu(self, gpu_idx):
        for domain_name, name in self.broadcast_name:
            self.get_broadcast(domain_name, name, gpu_idx)

    # memory bank calculation function
    @torch.no_grad()
    def updated_new_data_memory(self, domain, indices, outputs):
        """Compute new memory bank in indices by momentum

        Args:
            indices: indices of memory bank features to update
            outputs: output of features
            domain (str): 'source', 'target'
        """
        memory_bank = self.get_attr(domain, "memory_bank")
        data_memory = torch.index_select(memory_bank, 0, indices)

        outputs = F.normalize(outputs, dim=1)
        m = self.m
        new_data_memory = data_memory * m + (1 - m) * outputs
        return F.normalize(new_data_memory, dim=1)

    def _get_Z(self, domain, vec, t):
        """Get denominator in ID

        Args:
            vec: output features [batch_size, dim]
            domain (str): 'source', 'target'

        Returns:
            [batch_size] denominator in ID
        """
        bank = self.get_attr(domain, "memory_bank")  # [data_len]
        Z = torchutils.contrastive_sim_z(vec, bank, tao=t)
        return Z

    def _get_all_dot_products(self, domain, vec):
        """get dot product with all vectors in memory bank

        Args:
            vec: [bs, dim]
            domain (str): 'source', 'target'

        Returns:
            [bs, data_len]
        """
        assert len(vec.size()) == 2
        bank = self.get_attr(domain, "memory_bank")
        return torch.matmul(vec, torch.transpose(bank, 1, 0))

    def _compute_I2C_loss(self, batch_feat, centroids, t=0.05):
        """Loss CrossSelf in essay (Cross-domain Instance-Prototype SSL)

        Args:
            domain (str): 'source', 'target'
            loss_type (str, optional): 'each', 'all', 'src', 'tgt'. Defaults to "zero".
        """
        loss = torch.Tensor([0]).cuda()

        phi = t

        p = torchutils.contrastive_sim(batch_feat, centroids, tao=phi)
        z = torch.sum(p, dim=-1)  # [bs]
        p = p / z.unsqueeze(1)  # [bs, k]

        cur_loss = -torch.sum(p * torch.log(p)) / batch_feat.size(0)

        loss = loss + cur_loss

        return loss


    def _compute_proto_loss(self, batch_feat, batch_labels, centroids, t=0.05):
        """Loss PC in essay (part of In-domain Prototypical Contrastive Learning)

        Args:
            domain (str): 'source', 'target'
            loss_type (str, optional): 'each', 'all', 'src', 'tgt'. Defaults to "zero".
        """
        loss = torch.Tensor([0]).cuda()

        batch_centroids = centroids[batch_labels]

        dot_exp = torch.exp(
            torch.sum(batch_feat * batch_centroids, dim=-1) / t
        )

        assert not torch.isnan(batch_feat).any()
        assert not torch.isnan(batch_centroids).any()
        assert not torch.isnan(dot_exp).any()

        # calculate Z
        # all_phi = t if is_fix else phis.unsqueeze(0).repeat(outputs.shape[0], 1)
        z = torchutils.contrastive_sim_z(batch_feat, centroids, tao=t)

        # calculate loss
        p = dot_exp / z

        loss = loss - torch.sum(torch.log(p)) / p.size(0)

        # loss /= n_kmeans

        return loss
    
    def _compute_cluster_loss(self,preds,batch_labels,centroids):

        batch_centroids = centroids[batch_labels]

        loss = self.mse_loss(preds, batch_centroids)
        return loss

    def _compute_CD_loss(self, domain, loss_type, t=0.05):
        """Loss CDS in essay arXiv:2003.08264v1, not used in essay

        Args:
            domain (str): different domain from current one
            loss_type (str): 'cross'

        Returns:
            CD loss
        """
        assert loss_type in ["cross"]

        bank = self.get_attr(reverse_domain(domain), "memory_bank")
        if self.config.loss_params.sample_ratio:
            num_sample = int(self.config.loss_params.sample_ratio * len(bank))
            sampled_index = random.sample(list(range(len(bank))), num_sample)
            sampled_index = torch.tensor(sampled_index)
            bank = bank[sampled_index]

        # [bs, data_len], numerator of P_{i',i}^{s->t}
        prods = torchutils.contrastive_sim(self.outputs, bank, tao=t)
        # [bs]
        z = torch.sum(prods, dim=-1)
        # [bs, data_len] P_{i',i}^{s->t}
        p = prods / z.unsqueeze(1)
        aux = p.max(dim=1)[1]
        # double sum
        loss = -torch.sum(p * torch.log(p)) / self.batch_size
        return loss[None,], aux

    def _compute_ID_loss(self, domain, loss_type, t=0.05):
        """Loss ID (Instance Discrimination), not used in essay.

        Args:
            domain (str): 'source', 'target'
            loss_type (str): 'each', 'all'

        Returns:
            ID loss
        """
        assert loss_type in ["all", "each", "src", "tgt"]

        loss = torch.Tensor([0]).cuda()
        if loss_type == "src":
            if domain == "target":
                return loss
            else:
                clus = "each"
        if loss_type == "tgt":
            if domain == "source":
                return loss
            else:
                clus = "each"

        bank = self.get_attr(domain, "memory_bank")

        memory_vecs = torch.index_select(bank, 0, self.indices)
        prods = torch.sum(memory_vecs * self.outputs, dim=-1)
        # [bs], numerator of P_i^s
        prods_exp = torch.exp(prods / t)
        # [bs], denominator of P_i^s

        if self.config.loss_params.sample_ratio:
            num_sample = int(self.config.loss_params.sample_ratio * len(bank))
            sampled_index = random.sample(list(range(len(bank))), num_sample)
            sampled_index = torch.tensor(sampled_index)
            sampled_bank = bank[sampled_index]
            ratio_inv = 1 / self.config.loss_params.sample_ratio
        else:
            sampled_bank = bank
            ratio_inv = 1

        Z = torchutils.contrastive_sim_z(self.outputs, sampled_bank, tao=t)
        Z = ratio_inv * Z
        if loss_type == "all":
            bank_rev = self.get_attr(reverse_domain(domain), "memory_bank")
            Z = Z + torchutils.contrastive_sim_z(self.outputs, bank_rev, tao=t)
        # [bs], P_i^s
        p = prods_exp / Z
        loss = -torch.sum(torch.log(p)) / self.batch_size
        return loss[
            None,
        ]

    def _compute_loss(self, domain, loss_type=None, t=0.05):
        loss_name, loss_args = loss_type.split("-")
        loss_fn = getattr(self, f"_compute_{loss_name}_loss")
        aux = None
        if loss_name in ["CD"]:
            loss, aux = loss_fn(domain, loss_type=loss_args, t=t)
        else:
            loss = loss_fn(domain, loss_type=loss_args, t=t)

        assert not torch.isinf(loss).any()
        if loss < 0:
            print(loss)
            print(loss_name)
            assert loss >= 0

        return loss, aux


    def forward(self, indices, outputs, domain, gpu_idx):
        self.indices = indices.detach()
        self.batch_size = self.indices.size(0)
        self.outputs = outputs
        self._identify_gpu(gpu_idx)

        loss_part = []
        loss = torch.zeros(1).cuda()
        aux_list = {}
        for i, ls in enumerate(self.loss):
            if (
                self.epoch <= self.config.loss_params.start[i]
                or self.epoch >= self.config.loss_params.end[i]
                or ls.split("-")[0] in ["cls", "tgt", "semi", "norm"]
            ):
                l = torch.zeros(1).cuda()
            else:
                l, aux = self._compute_loss(domain, ls, self.t[i])
                aux_list[ls] = aux
            loss_part.append(l)
            loss = loss + l

        new_data_memory = self.updated_new_data_memory(domain, indices, outputs)

        return loss, new_data_memory, loss_part, aux_list

    def _calc_gradient_penalty(self, netD, real_data, fake_data, input_att):

        alpha = torch.rand(self.batch_size, 1).cuda()
        alpha = alpha.expand(real_data.size())
        lambda1 = 10

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = netD(torch.cat([interpolates, input_att],dim=-1))

        gradient_penalty = self.gp_loss(x=interpolates,y = disc_interpolates) * lambda1
        
        return gradient_penalty


    def forward_original(self, indices, outputs, domain, gpu_idx):
        self.indices = indices.detach()
        self.batch_size = self.indices.size(0)
        self.outputs = outputs
        self._identify_gpu(gpu_idx)

        loss_part = []
        loss = torch.zeros(1).cuda()
        aux_list = {}
        for i, ls in enumerate(self.loss):
            if (
                self.epoch <= self.config.loss_params.start[i]
                or self.epoch >= self.config.loss_params.end[i]
                or ls.split("-")[0] in ["cls", "tgt", "semi", "norm"]
            ):
                l = torch.zeros(1).cuda()
            else:
                l, aux = self._compute_loss(domain, ls, self.t[i])
                aux_list[ls] = aux
            loss_part.append(l)
            loss = loss + l

        new_data_memory = self.updated_new_data_memory(domain, indices, outputs)

        return loss, new_data_memory, loss_part, aux_list


def loss_info(feat, mb_feat, label, t=0.1):
    loss = torch.Tensor([0]).cuda()
    z = torchutils.contrastive_sim_z(feat, mb_feat, tao=t)
    for i, lb in enumerate(label):
        pos = mb_feat[label == lb]
        up = torch.exp(torchutils.dot(feat[i], pos) / t)
        p = up / z[i]
        assert all(p < 1)
        loss += -torch.sum(torch.log(p)) / len(p)
    loss /= len(feat)
    return loss


@torch.no_grad()
def update_data_memory(data_memory, outputs, m=0.9):
    outputs = F.normalize(outputs, dim=1)
    new_data_memory = data_memory * m + (1 - m) * outputs
    return F.normalize(new_data_memory, dim=1)


class GradientPaneltyLoss(nn.Module):
    def __init__(self):
        super(GradientPaneltyLoss, self).__init__()

    def forward(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones_like(y)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)
