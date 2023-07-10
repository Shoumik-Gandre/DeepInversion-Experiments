import torch.nn as nn


class LeNet5(nn.Module):

    def __init__(self, in_channels: int=1, num_labels: int=10):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(120)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_labels)

    def forward(self, img, out_feature=False):
        output = self.conv1(img)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        feature = self.flatten(output) # .view(-1, 120) the shape[1] of feature has to be 120
        output = self.fc1(feature)
        output = self.bn4(output)
        output = self.relu4(output)
        output = self.fc2(output)
        if out_feature == False:
            return output
        else:
            return output,feature
    