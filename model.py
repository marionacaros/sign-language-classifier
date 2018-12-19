import torch.nn as nn

### Create the model

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size):
        super(Unit, self).__init__()

        pad = int((filter_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=filter_size, stride=1,
                              padding=pad)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output



class CNN(nn.Module):

    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.unit1 = Unit(in_channels=1, out_channels=8, filter_size=3)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # output of each feature map is now 28/2 = 14

        self.unit2 = Unit(in_channels=8, out_channels=32, filter_size=5)
        # Output size of each of the 32 feature maps remains 14

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # After max pooling, output of each feature map = 14/2 = 7

        # Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.maxpool1, self.unit2, self.maxpool2)

        self.relu = nn.ReLU(inplace=True)

        # Flatten the feature maps
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(32 * 7 * 7, 600)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(600, num_classes)

    def forward(self, x):  # Building a graph
        out = self.net(x)

        # Flatten the output it will take the shape (batch_size, num_features)
        num_feat = 32 * 7 * 7
        out = out.view(-1, num_feat)  # reshape
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out



