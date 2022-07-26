from curses import raw
import torch
import torch.nn as nn
import os

import torch.nn.functional as F


from torchvision.models.resnet import BasicBlock, Bottleneck, model_urls

import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Normalize(nn.Module):
 
    def forward(self, x):
        return F.normalize(x, dim=-1)

class Network(nn.Module):
    def __init__(self, params, out_dim = 200):
        super().__init__()

        self.params = params

        # model_type = 'resnet18'
        backbone = self.params.backbone

        model_dir = os.environ['WORK'] + '/birds/.cache/models'
        latent_dim = self.params.latent_dim

        if backbone == 'resnet18':
            self.feature = resnet18(pretrained=True,model_dir=model_dir)
        elif backbone == 'resnet34':
            self.feature = resnet34(pretrained=True,model_dir=model_dir)
        elif backbone == 'resnet50':
            self.feature = resnet50(pretrained=True,model_dir=model_dir)
            latent_dim = 2048
        elif backbone == 'resnet101':
            self.feature = resnet101(pretrained=True,model_dir=model_dir)
            latent_dim = 2048

        if self.params.freeze_backbone:
            for param in self.feature.parameters():
                param.requires_grad = False

        self.is_atn_pool = self.params.is_pool_source_keypoints
        n_mult_feat_dim = 1
        if self.is_atn_pool:
            self.atn_pool = BAP(pool='GAP',non_lin='no')

            self.n_attention = 4
            self.attentions = nn.Sequential(
                nn.Conv2d(in_channels=latent_dim, out_channels=512, kernel_size=(3, 3), padding=1, stride=1,
                          bias=True),
                nn.ReLU(True),
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(in_channels=256, out_channels=self.n_attention, kernel_size=(3, 3), padding=1, stride=1,
                          bias=True))
            n_mult_feat_dim = n_mult_feat_dim *(self.n_attention +1)

        self.fc = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(latent_dim*n_mult_feat_dim, out_dim),
        )
    
    def forward(self, x, atn_maps = None):
        feat_vector, feat_map = self.feature(x)       # Sequential_1
        if self.is_atn_pool:
            feat_vector = self.forward_atn(feat_map,atn_maps)
        pred = self.fc(feat_vector)       # Dropout -> Dense -> Activation
        return pred, feat_vector

    def forward_atn(self, feat_map, atn_maps):
        if atn_maps is None:
            atn_maps = self.attentions(feat_map)
            atn_maps = torch.sigmoid(atn_maps)
        else:
            atn_maps = F.adaptive_max_pool2d(atn_maps, output_size=(feat_map.shape[2], feat_map.shape[3]))
        
        # add global pool feat vector
        atn_ones = torch.ones_like(atn_maps[:,0]).unsqueeze(1)
        atn_maps = torch.cat((atn_ones,atn_maps),dim=1)
        feat_vector, _ = self.atn_pool(feat_map,atn_maps)
        return feat_vector

# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP', non_lin='sigmoid'):
        super(BAP, self).__init__()
        self.non_lin = non_lin
        # self.n_supervised = n_supervised
        self.softmax2d = nn.Softmax2d()
        # self.grad_damp = grad_damp(alpha=0.1)
        assert pool in ['GAP', 'GMP']
        assert non_lin in ['sigmoid','no']
        if pool == 'GAP':
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, raw_attentions, non_lin=None):

        if non_lin is None:
            non_lin = self.non_lin
        B = features.size(0)
        M = raw_attentions.size(1)
        if non_lin == 'sigmoid':
            attentions = torch.sigmoid(raw_attentions)
        else:
            attentions = raw_attentions

        feature_matrix = []
        for i in range(M):
            AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
            AiF = F.normalize(AiF)
            feature_matrix.append(AiF)
        feature_matrix = torch.cat(feature_matrix, dim=-1)

        return feature_matrix, attentions



class ResNet(models.ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):

        super().__init__(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_prepool = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x, x_prepool





def resnet18(pretrained=False, model_dir=None, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=model_dir))
    return model

def resnet34(pretrained=False, model_dir=None, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir=model_dir))
    return model


def resnet50(pretrained=False, model_dir=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir=model_dir))
    return model


def resnet101(pretrained=False, model_dir=None,**kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir=model_dir))
    return model