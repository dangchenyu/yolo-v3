import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=3, padding=1)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = out + input
        return out


class ZhazhaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ("0", nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)),
            ("1", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2))
        ]))
        self.res1 = nn.Sequential(OrderedDict([
            ('0', ResBlock(in_channels=64)),
            ('conv', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2))
        ]))
        self.res2 = nn.Sequential(OrderedDict([
            ('0', ResBlock(in_channels=128)),
            ('1', ResBlock(in_channels=128)),
            ('conv', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2))
        ]))
        self.res3 = nn.Sequential(OrderedDict([
            ('0', ResBlock(in_channels=256)),
            ('1', ResBlock(in_channels=256)),
            ('2', ResBlock(in_channels=256)),
            ('3', ResBlock(in_channels=256)),
            ('4', ResBlock(in_channels=256)),
            ('5', ResBlock(in_channels=256)),
            ('6', ResBlock(in_channels=256)),
            ('7', ResBlock(in_channels=256))
        ]))
        self.res3_out = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2)
        self.res4 = nn.Sequential(OrderedDict([
            ('0', ResBlock(in_channels=512)),
            ('1', ResBlock(in_channels=512)),
            ('2', ResBlock(in_channels=512)),
            ('3', ResBlock(in_channels=512)),
            ('4', ResBlock(in_channels=512)),
            ('5', ResBlock(in_channels=512)),
            ('6', ResBlock(in_channels=512)),
            ('7', ResBlock(in_channels=512))
        ]))
        self.res4_out = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2)
        self.res5 = nn.Sequential(OrderedDict([
            ('0', ResBlock(in_channels=1024)),
            ('1', ResBlock(in_channels=1024)),
            ('2', ResBlock(in_channels=1024)),
            ('3', ResBlock(in_channels=1024)),
            ('conv0_a', nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)),
            ('conv0_b', nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)),
            ('conv1_a', nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)),
            ('conv1_b', nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)),
            ('conv2_a', nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)),
            ('conv2_b', nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1))
        ]))
        self.conv_m = nn.Sequential(OrderedDict([
            ('conv0_a', nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1)),
            ('conv0_b', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)),
            ('conv1_a', nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)),
            ('conv1_b', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)),
            ('conv2_a', nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)),
            ('conv2_b', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)),
        ]))
        self.conv_l = nn.Sequential(OrderedDict([
            ('conv0_a', nn.Conv2d(in_channels=384, out_channels=128, kernel_size=1)),
            ('conv0_b', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)),
            ('conv1_a', nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)),
            ('conv1_b', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)),
            ('conv2_a', nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)),
            ('conv3_b', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)),
        ]))
        self.mapping_s = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.mapping_m = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1)
        self.detection_s = nn.Conv2d(in_channels=1024, out_channels=255, kernel_size=1)
        self.detection_m = nn.Conv2d(in_channels=512, out_channels=255, kernel_size=1)
        self.detection_l = nn.Conv2d(in_channels=256, out_channels=255, kernel_size=1)

    def forward(self, input):
        out = self.conv1(input)
        out = self.res1(out)
        out = self.res2(out)
        out_l = self.res3(out)
        out = self.res3_out(out_l)
        out_m = self.res4(out)
        out = self.res4_out(out_m)
        out_s = self.res5(out)
        result_s = self.detection_s(out_s)
        out = self.mapping_s(out)
        out = F.upsample(out, scale_factor=2)
        out = torch.cat((out, out_m), dim=1)
        out = self.conv_m(out)
        result_m = self.detection_m(out)
        out = self.mapping_m(out)
        out = F.upsample(out, scale_factor=2)
        out = torch.cat((out, out_l), dim=1)
        out = self.conv_l(out)
        result_l = self.detection_l(out)
        return result_s, result_m, result_l


if __name__ == '__main__':
    img = torch.randn(8, 3, 512, 512)
    darknet = ZhazhaNet()
    out = darknet(img)
    print(out[-1].shape)
