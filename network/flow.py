import torch
from torch import nn


class CNN(nn.Module):
    ''' E_step_z with Residue Learning'''
    def __init__(self, depth=17, n_channels=64, in_chan=2, out_chan=2, kernel_size = 3, padding = 1):
        super(CNN, self).__init__()

        layers = []
        self.in_chan = in_chan

        layers.append(nn.Conv2d(in_channels=in_chan, out_channels=n_channels, kernel_size=kernel_size, padding=padding))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=out_chan, kernel_size=kernel_size, padding=padding, bias=False))
        self.cnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x, y):
        input = torch.cat((x,y), dim = 1)
        output = self.cnn(input)
        return output

    def _initialize_weights(self):
        import torch.nn.init as init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)