import torch
import torch.nn as nn


class unetpad1(nn.Module):
    
    def __init__(
        self, num_classes=2, in_channels=3, is_deconv=False, is_batchnorm=False, *args, **kwargs):
        super(unetpad1, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        filters = [64, 128]

        self.down1 = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.center = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.up1 = unetUp(filters[1] + filters[0], filters[0], self.is_deconv)
        self.final = nn.Conv2d(filters[0], num_classes, 1)

    def forward(self, inputs):
        x, befdown1 = self.down1(inputs)
        x = self.center(x)
        x = self.up1(befdown1, x)

        return self.final(x)


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1),
                nn.ReLU(),
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs1 = self.down(outputs)
        return outputs1, outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, 2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        return self.conv(torch.cat([inputs1, outputs2], 1))
