import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet_Model(nn.Module): ######TRANSFER LEARNING with resnet 50 as base
    def __init__(self, num_classes):
        super(ResNet_Model, self).__init__()

        self.backbone = resnet50(pretrained = True)#######load old pretrained weights

        for param in self.backbone.parameters():
            param.requires_grad = False  # freezing

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

        feature_dim = self.backbone.fc.in_features #get the number of connections entering

        self.backbone.fc = nn.Identity() ## reference below its just used as aplaceholder 

        #place two heads one for class other for bounding boxes
        self.class_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        features = self.backbone(x)         
        class_out = self.class_head(features)
        bbox_out = self.bbox_head(features)
        return class_out, bbox_out

########REFERENCES: TRANSFER LEARNING

# https://github.com/tanjeffreyz/yolo-v1/blob/main/models.py
# https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch