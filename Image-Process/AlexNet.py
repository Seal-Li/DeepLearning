# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torchvision import transforms, datasets, utils
import torch.optim as optim

# 全局取消证书安全认证，否则数据集下载会因验证报错
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

"""
1 image size: c * w * w
2 filter size: f * f
3 stride: s
4 padding: p
then the size of image after Convolution:
n = ((w - f + 2p) / s) + 1
"""

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2), # input (3 224, 224), output (48, 55, 55)
            nn.ReLu(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # output (48, 27, 27)
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, padding=2), # output (128, 27, 27)
            nn.ReLu(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # output (128, 13, 13)
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, padding=1), # output (192, 13, 13)
            nn.ReLu(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1), # output (192, 13, 13)
            nn.ReLu(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1), # output (128, 13, 13)
            nn.ReLu(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # output (128, 6, 6)
        )
        
        # Dropout was used betwwen fully connected layers usually 
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, start_dim=1)
            x = self.classifier(x)
            return x

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    
    transform = {
        "train":transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val":transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
    
    net = AlexNet()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()