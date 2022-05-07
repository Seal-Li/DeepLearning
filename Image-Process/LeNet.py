# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

# 全局取消证书安全认证，否则数据集下载会因验证报错
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # images size is (3, 32, 32)--(channels, height, width)
        self.conv1 = nn.Conv2d(3, 6, 5) # (in_channels, out_channels, kernel_size), output size is (6, 28, 28)
        self.pool = nn.MaxPool2d(2, 2) # kernel_size, ouput size is (6, 14, 14)
        self.conv2 = nn.Conv2d(6, 16, 5) # output size is (16, 10, 10)
        self.fc1 = nn.Linear(16*5*5, 120) # (in_size, out_size), Limear flatten the image tensor to an one dimension vector
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    
    def forward(self, x):
        # first layer
        x = self.pool(F.relu(self.conv1(x)))
 
        # second layer
        x = self.pool(F.relu(self.conv2(x)))
        
        # flatten x
        x = x.view(-1, 16*5*5)
        
        # first fully connected layer
        x = F.relu(self.fc1(x))
        
        # second fully connected layer 
        x = F.relu(self.fc2(x))
        
        # third fully connected
        x = self.fc3(x)
        
        return x

if __name__ == '__main__':
    # loading dataset and preprocess
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # transform images to tensor and normalized

    # if you have download the dataset already, then set the parameter of download as False, otherwise True
    trainset = torchvision.datasets.CIFAR10(download=False, root="./data", train=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(download=False, root="./data", train=False, transform=transform)

    # setting the batch size, wether you shuffle the order of images, and number works if you device support multi-thread
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=0)

    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck") # classes of CIFAR10

    # train setting
    net = LeNet()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # train
    for epoch in range(2):
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            optimizer.zero_grad() # set the grad is 0, otherwise the grad will accumulate
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward() # backward to adjust parameters
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"epoch: {epoch+1}, \ti: {i+1}, \trunning_loss: {running_loss/2000}")
                running_loss = 0.0
    print("Finished Training")
    
    # test
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict==labels).sum().item()
    print(f"Accuracy:{correct / total}%")