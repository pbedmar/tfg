import torch
from torch import nn
from torch.nn import functional as F

# Based on K. Simonyan and A. Zisserman paper Very Deep Convolutional Networks for Large-Scale Image Recognition, 2014.
# Implementation obtained from medium on 6/5/22: https://medium.com/@tioluwaniaremu/vgg-16-a-simple-implementation-using-pytorch-7850be4d14a1
# and adapted for 256x256 images
class VGG16(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), stride=(1,1))

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3,3), padding=(1,1), stride=(1,1))

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1,1), stride=(1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1,1), stride=(1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1,1), stride=(1, 1))

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1,1), stride=(1, 1))
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1), stride=(1, 1))
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1), stride=(1, 1))

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1), stride=(1, 1))
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

        self.fc1 = nn.Linear(32768, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):

        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=(2, 2), dilation=(1, 1))

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=(2, 2), dilation=(1, 1))

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=(2, 2), dilation=(1, 1))

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=(2, 2), dilation=(1, 1))

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=(2, 2), dilation=(1, 1))

        x = x.reshape(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)

        x = F.relu(self.fc3(x))
        x = F.dropout(x, 0.5)

        x = self.fc4(x)     # Use sigmoid to keep output between 0 and 1. if we were doing multiclass logistic
                            # regression, we would use softmax activation function
                            # 1 output unit -> F.sigmoid() + torch.nn.BCELoss == torch.nn.BCEWithLogitsLoss
                            # 2 output units -> torch.nn.CrossEntropyLoss() (softmax included)
        return x
