import torch


# Model with 2 heads for overclustering
class IICNet(torch.nn.Module):
    def __init__(self, config):
        super(IICNet, self).__init__()
        self.in_channels = config.in_channels
        self.pad = config.pad
        self.conv_size = config.conv_size
        self.features = self._make_layers()
        self.head_a = IICNetHead(config.out_channels_a)
        self.head_b = IICNetHead(config.out_channels_b)

    def _make_layers(self):
        layers = []

        # Structure taken from supplementary, base C
        layers += self.conv_block(self.in_channels, 64, 1)
        layers += self.conv_block(64, 128, 1)
        layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        layers += self.conv_block(128, 256, 1)
        layers += self.conv_block(256, 256, 1)
        layers += self.conv_block(256, 512, 2)
        layers += self.conv_block(512, 512, 2)
        return torch.nn.Sequential(*layers)

    def forward(self, x, head=0):
        # Forward x to all layers and one head
        x = self.features(x)
        if head == 0:
            x = self.head_a(x)
        elif head == 1:
            x = self.head_b(x)
        # Scale image
        return torch.nn.functional.interpolate(x, size=128, mode="bilinear", align_corners=False)

    def conv_block(self, in_channels, out_channels, dilation):
        # Create one convolutional block consisting of a convolution, a batch normalization and a ReLu layer
        return [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=self.conv_size, stride=1,
                                padding=self.pad, dilation=dilation, bias=False),
                torch.nn.BatchNorm2d(out_channels, track_running_stats=False), torch.nn.ReLU(inplace=True)]


class IICNetHead(torch.nn.Module):
    def __init__(self, out_channels):
        super(IICNetHead, self).__init__()
        # Create the head, which only differs in number of output channels
        self.features = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=1,
                                    stride=1, dilation=1, padding=1, bias=False),
                    torch.nn.Softmax2d())

    def forward(self, x):
        return self.features(x)
