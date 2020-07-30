import torch.nn as nn
import torch.nn.functional as  F

class ForLayerConvNet(nn.Module):
    """
    define a 4 layers convolution network
    """
    def __init__(self, output_size=10):
        super(ForLayerConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3), stride=1, padding=1) # padding=1 keep the size
        self.conv2 = nn.Conv2d(32, 64, (3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2*2*64, 120)
        self.fc2 = nn.Linear(120, output_size)

    def forward(self, x):
        """
        define the forward path
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 2*2*64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x