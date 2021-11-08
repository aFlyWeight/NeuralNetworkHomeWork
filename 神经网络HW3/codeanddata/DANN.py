import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function


class ReverseLayer(Function):
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class FeatureExtractor(nn.Module):
    def __init__(self, D_in, Hidden,D_out):
        super(FeatureExtractor, self).__init__()
        feature = nn.Sequential()
        feature.add_module('f_linear1', nn.Linear(D_in, Hidden))
        feature.add_module('f_bn1', nn.BatchNorm1d(Hidden))
        feature.add_module('f_relu1', nn.ReLU(True)) 
        feature.add_module('f_linear2', nn.Linear(Hidden, D_out))
        feature.add_module('f_bn2', nn.BatchNorm1d(D_out))
 #       feature.add_module('f_drop1', nn.Dropout2d())
        feature.add_module('f_relu2', nn.ReLU(True))
        self.feature = feature

    def forward(self, x):
        return self.feature(x)


class Classifier(nn.Module):
    def __init__(self, D_in, Hidden1, Hidden2, D_out):
        super(Classifier, self).__init__()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fl1', nn.Linear(D_in, Hidden1))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(Hidden1))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
#       self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fl2', nn.Linear(Hidden1, Hidden2))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(Hidden2))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        
        self.class_classifier.add_module('c_fl3', nn.Linear(Hidden2, D_out))
#        self.class_classifier.add_module('c_bn3', nn.BatchNorm1d(Hidden2))
        self.class_classifier.add_module('c_relu3', nn.ReLU(True))
        

    def forward(self, x):
        return self.class_classifier(x)


class Domain_Classifier(nn.Module):
    def __init__(self, D_in, Hidden1, Hidden2, D_out):
        super(Domain_Classifier, self).__init__()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fl1', nn.Linear(D_in, Hidden1))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(Hidden1))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
 #       self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fl2', nn.Linear(Hidden1, Hidden2))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(Hidden2))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        
        self.class_classifier.add_module('c_fl3', nn.Linear(Hidden2, D_out))
 #       self.class_classifier.add_module('c_bn3', nn.BatchNorm1d(Hidden2))
        self.class_classifier.add_module('c_relu3', nn.ReLU(True))
    def forward(self, x):
        return self.class_classifier(x)





class DANN(nn.Module):

    def __init__(self,D_in, F_Hidden1,F_out,C_Hidden1,C_Hidden2,C_out,DC_Hidden1,DC_Hidden2,DC_out):
        super(DANN, self).__init__()
        self.feature = FeatureExtractor(D_in, F_Hidden1, F_out)
        self.classifier = Classifier(F_out,C_Hidden1,C_Hidden2,C_out)
        self.domain_classifier = Domain_Classifier(F_out,DC_Hidden1,DC_Hidden2,DC_out)
        self.ReverseLayer = ReverseLayer()
#        self.device = device

    def forward(self, input_data, alpha=1):
        feature = self.feature(input_data)
        class_output = self.classifier(feature)
        feature = self.ReverseLayer.apply(feature,alpha)
        domain_output = self.domain_classifier(feature)
        return class_output, domain_output

    '''
    def get_adversarial_result(self, x, source=True, alpha=1):
        loss_fn = nn.BCELoss()
        if source:
            domain_label = torch.ones(len(x)).long().to(self.device)        #域标签
        else:
            domain_label = torch.zeros(len(x)).long().to(self.device)
        x = ReverseLayer.apply(x, alpha)
        domain_pred = self.domain_classifier(x)
        loss_adv = loss_fn(domain_pred, domain_label.float())
        return loss_adv
    '''
