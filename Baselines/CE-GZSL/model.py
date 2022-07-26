import torch.nn as nn
import torch
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def reparameter(mu,sigma):
    return (torch.randn_like(mu) *sigma) + mu

class Embedding_Net(nn.Module):
    def __init__(self, opt):
        super(Embedding_Net, self).__init__()

        self.fc1 = nn.Linear(opt.resSize, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, opt.outzSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, features):
        embedding= self.relu(self.fc1(features))
        out_z = F.normalize(self.fc2(embedding), dim=1)
        return embedding,out_z


class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class MLP_CRITIC(nn.Module):
    def __init__(self, opt):
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h


class Dis_Embed_Att(nn.Module):
    def __init__(self, opt):
        super(Dis_Embed_Att, self).__init__()
        self.fc1 = nn.Linear(opt.embedSize+opt.attSize, opt.nhF)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.nhF, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, input):
        h = self.lrelu(self.fc1(input))
        h = self.fc2(h)
        return h


from torchvision import models


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

    model_ft = nn.Sequential(*list(model_ft.children())[:-1])
    
    return model_ft, input_size
