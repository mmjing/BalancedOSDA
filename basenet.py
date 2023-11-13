from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import numpy as np

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=1000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)        


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

class VGGBase(nn.Module):
    # Model VGG
    def __init__(self, feat_size=2048):
        super(VGGBase, self).__init__()
        model_ft = models.vgg19(pretrained=True)
        # print(model_ft)
        mod = list(model_ft.features.children())
        self.lower = nn.Sequential(*mod)
        mod = list(model_ft.classifier.children())
        mod.pop()
        # print(mod)
        self.upper = nn.Sequential(*mod)
        self.linear1 = nn.Linear(4096, feat_size)
        self.bn1 = nn.BatchNorm1d(feat_size, affine=True)
        self.linear2 = nn.Linear(feat_size, feat_size)
        self.bn2 = nn.BatchNorm1d(feat_size, affine=True)

    def forward(self, x, target=False):
        x = self.lower(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.upper(x)
        x = F.dropout(F.leaky_relu(self.bn1(self.linear1(x))), training=False)
        x = F.dropout(F.leaky_relu(self.bn2(self.linear2(x))), training=False)
        if target:
            return x
        else:
            return x
        return x


class AlexBase(nn.Module):
    def __init__(self):
        super(AlexBase, self).__init__()
        model_ft = models.alexnet(pretrained=True)
        mod = []
        print(model_ft)
        for i in range(18):
            if i < 13:
                mod.append(model_ft.features[i])
        mod_upper = list(model_ft.classifier.children())
        mod_upper.pop()
        # print(mod)
        self.upper = nn.Sequential(*mod_upper)
        self.lower = nn.Sequential(*mod)
        self.linear1 = nn.Linear(4096, 100)
        self.bn1 = nn.BatchNorm1d(100, affine=True)
        self.linear2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)

    def forward(self, x, target=False, feat_return=False):
        x = self.lower(x)
        x = x.view(x.size(0), 9216)
        x = self.upper(x)
        feat = x
        x = F.dropout(F.leaky_relu(self.bn1(self.linear1(x))))
        x = F.dropout(F.leaky_relu(self.bn2(self.linear2(x))))
        if feat_return:
            return feat
        if target:
            return x
        else:
            return x


class Classifier(nn.Module):
    def __init__(self, num_classes=12,feat_size=1000):
        super(Classifier, self).__init__()
        # self.fc1 = nn.Linear(100, 100)
        # self.bn1 = nn.BatchNorm1d(100, affine=True)
        # self.fc2 = nn.Linear(100, 100)
        # self.bn2 = nn.BatchNorm1d(100, affine=True)
        self.fc3 = nn.Linear(feat_size, num_classes)  # nn.Linear(100, num_classes)

    # def set_lambda(self, lambd):
    #     self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False, iter_num=10):
        if reverse:
            x = grad_reverse(x, calc_coeff(iter_num))
        x = self.fc3(x)
        return x

class CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim = 500):
        super(CLS, self).__init__()
        if bottle_neck_dim:
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.fc = nn.Linear(bottle_neck_dim, out_dim)
            self.main = nn.Sequential(
                self.bottleneck,
                nn.Sequential(
                    nn.BatchNorm1d(bottle_neck_dim),
                    nn.LeakyReLU(0.2, inplace = True),
                    self.fc
                ),
                nn.Softmax(dim = -1)
            )
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(
                self.fc,
                nn.Softmax(dim = -1)
            )

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out

class ResBase(nn.Module):
    def __init__(self, option='resnet18', pret=True, feat_size=100):
        super(ResBase, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(*mod)
        # default unit size 100
        self.linear1 = nn.Linear(2048, feat_size)
        self.bn1 = nn.BatchNorm1d(feat_size, affine=True)
        self.linear2 = nn.Linear(feat_size, feat_size)
        self.bn2 = nn.BatchNorm1d(feat_size, affine=True)
        self.linear3 = nn.Linear(feat_size, feat_size)
        self.bn3 = nn.BatchNorm1d(feat_size, affine=True)
        self.linear4 = nn.Linear(feat_size, feat_size)
        self.bn4 = nn.BatchNorm1d(feat_size, affine=True)
    def forward(self, x,reverse=False):
        x = self.features(x)
        x = x.view(x.size(0), self.dim)
        # best with dropout
        if reverse:
            x = x.detach()
        x = F.dropout(F.relu(self.bn1(self.linear1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.linear2(x))), training=self.training)
        return x

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}

class ResNetFc(nn.Module):
  def __init__(self, resnet_name, bottleneck_dim=256):
    super(ResNetFc, self).__init__()
    model_resnet = resnet_dict[resnet_name](pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

    self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
    self.bottleneck.apply(init_weights)
    self.__in_features = bottleneck_dim

  def forward(self, x):
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    x = self.bottleneck(x)
    return x

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}]
    return parameter_list

vgg_dict = {"VGG11":models.vgg11, "VGG13":models.vgg13, "VGG16":models.vgg16, "VGG19":models.vgg19, "VGG11BN":models.vgg11_bn, "VGG13BN":models.vgg13_bn, "VGG16BN":models.vgg16_bn, "VGG19BN":models.vgg19_bn} 
class VGGFc(nn.Module):
  def __init__(self, vgg_name, bottleneck_dim=256):
    super(VGGFc, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)

    self.bottleneck = nn.Linear(4096, bottleneck_dim)
    self.bottleneck.apply(init_weights)
    self.__in_features = bottleneck_dim

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    x = self.bottleneck(x)
    return x

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    parameter_list = [{"params":self.features.parameters(), "lr_mult":1, 'decay_mult':2}, \
                    {"params":self.classifier.parameters(), "lr_mult":1, 'decay_mult':2}, \
                    {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}]
    return parameter_list



class ResClassifier(nn.Module):
    def __init__(self, num_classes=12, feat_size=1000):
        super(ResClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feat_size, num_classes)
        )
        
    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False, iter_num=10):
        if reverse:
            x = grad_reverse(x, calc_coeff(iter_num))
        x = self.classifier(x)
        return x

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(weights_init)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x, reverse=True):
    if self.training:
        self.iter_num += 1
    x = x * 1.0
    if reverse:
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    # y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]


# from fclswgan
class MLP_CRITIC(nn.Module):
    def __init__(self, in_size, hidden_size): 
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h

# from Semi-supervised Domain Adaptation via Minimax Entropy (ICCV 2019)
class Predictor_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out
