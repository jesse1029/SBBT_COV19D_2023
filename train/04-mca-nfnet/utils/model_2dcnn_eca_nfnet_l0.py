import torch.nn as nn, torch.nn.functional as F
import timm
import torch
from timm.models.efficientnet import efficientnet_b3a, tf_efficientnet_b4_ns, tf_efficientnetv2_s, tf_efficientnetv2_m

from timm.models.convnext import *
#use tf_efficientnet_b4_ns


class eca_nfnet_l0(nn.Module):
    def __init__(self, n_classes, pretrained=True):

        super(eca_nfnet_l0, self).__init__()

        self.model = timm.create_model("hf_hub:timm/eca_nfnet_l0", pretrained=True)
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, 1)
        
        
    def forward(self, x):
        x = self.model(x)
        return x
'''    
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #to do
        e = tf_efficientnet_b4_ns()
        e = efficientnet_b3a(pretrained=True, drop_rate=0.3, drop_path_rate=0.2)
        
        self.b0 = nn.Sequential(
            e.conv_stem,
            e.bn1,
            #e.act1,
        )
        self.b1 = e.blocks[0]
        self.b2 = e.blocks[1]
        self.b3 = e.blocks[2]
        self.b4 = e.blocks[3]
        self.b5 = e.blocks[4]
        self.b6 = e.blocks[5]
        self.b7 = e.blocks[6]
        self.b8 = nn.Sequential(
            e.conv_head, #384, 1536
            e.bn2,
            #e.act2,
        )

        
        
        self.emb = nn.Linear(1536,224)
        self.logit = nn.Linear(224,1)
        

    def forward(self, image):
        batch_size = len(image)
        x = 2*image-1     

        x = self.b0(x) 
        x = self.b1(x) 
        x = self.b2(x)
        x = self.b3(x) 
        x = self.b4(x) 
        x = self.b5(x) 
        x = self.b6(x) 
        x = self.b7(x) 
        x = self.b8(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)

        x = self.emb(x)
        logit = self.logit(x)
     
        return logit


'''


def criterion(outputs, labels, gpu=None):
    if gpu!=None:
        return nn.BCEWithLogitsLoss().cuda(gpu)(outputs, labels)
    else:
        return nn.BCEWithLogitsLoss()(outputs, labels)