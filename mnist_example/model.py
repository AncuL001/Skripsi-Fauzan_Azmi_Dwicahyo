import torch
import torch.nn.functional as F
import torch.nn as nn


class CNNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, **kwargs):
        super(CNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16 * 14 * 14, out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.softmax(self.fc1(x), dim=1)

        return x

def test():
    model = CNNet(in_channels=1, out_channels=10)
    x = torch.randn(64, 1, 28, 28)
    print(model(x).shape)
    
if __name__ == "__main__":
    test()