import torch
import torch.nn as nn
import math


# Conv Bn Relu
class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)

        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


# Ours_DSAConv
class DSSConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DSSConv, self).__init__()
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, bias=False, groups=in_channels, kernel_size=3,
                      padding=1, stride=stride),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.attention_net = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )
        self.pw_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        x = self.dw_conv(x)
        global_feat = self.global_pool(x)
        global_feat = global_feat.view(x.size()[0], -1)
        attention_map = self.attention_net(global_feat)
        # reshape to shape of x for element-wise multiplication
        attention_map = attention_map.view(x.size()[0], x.size()[1], 1, 1)
        x = x + x * attention_map.expand_as(x)
        x = self.pw_conv(x)
        return x


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()  # 这里是串联支路
        self.stride = stride  # 步长控制

        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )  # 2/N
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)  # 对应于avg部分支路
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))  # block1,1×1卷积,将通道调整为1/2 N
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    DSSConv(out_planes // 2, out_planes // 2, stride=stride))  # 如果block num = 2 就会提前终止了, 这里pass掉
            elif idx == 1 and block_num > 2:
                self.conv_list.append(DSSConv(out_planes // 2, out_planes // 4, stride=stride))  # 3×3卷积，通道调整为1/4 N
            elif idx < block_num - 1:
                self.conv_list.append(
                    DSSConv(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))  # 这里 1/8 N
            else:
                self.conv_list.append(
                    DSSConv(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))  # 这里 1/8 N

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)  # block1

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))  # 这里是单独进行下采样的
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)  # 跳跃连接
        out_list.insert(0, out1)
        out = torch.cat(out_list, dim=1)
        return out


class Conv_to_9(nn.Module):
    def __init__(self):
        super(Conv_to_9, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, stride=1)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x2, x3 = x2.chunk(2, dim=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        return torch.cat([x1, x2, x3], dim=1)


class enhanceNet(nn.Module):
    def __init__(self,mode="train"):
        super(enhanceNet, self).__init__()
        # self.downsample = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1),
        #                                 nn.BatchNorm2d(16),
        #                                 nn.ReLU(inplace=True)) #2x
        self.bottleneck1 = nn.Sequential(CatBottleneck(in_planes=3, out_planes=24, stride=1),
                                         CatBottleneck(in_planes=24, out_planes=24, stride=1),
                                         )
        self.bottleneck2 = nn.Sequential(CatBottleneck(in_planes=3, out_planes=24, stride=1),
                                         CatBottleneck(in_planes=24, out_planes=24, stride=1),
                                         )
        self.conv_to_9 = Conv_to_9()
        self.conv_to = Conv_to_9()
        self.mode=mode
        # self.spatial_at = SpatialAttention()

    def muti_x_pow(self, x, r1, r2, r3):
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        return x

    def forward(self, x):

        out = self.bottleneck1(x)
        out = self.conv_to_9(out)
        r1, r2, r3 = torch.split(torch.nn.functional.tanh(out), 3, dim=1)
        enhance_image = self.muti_x_pow(x, r1, r2, r3)  # 获得了一次增强的图片
        if self.mode=="train":
            r_1 = torch.cat([r1, r2, r3], 1)

            out = self.bottleneck2(enhance_image)
            out = self.conv_to(out)
            r1, r2, r3 = torch.split(torch.nn.functional.tanh(out), 3, dim=1)
            enhance_image_stage2 = self.muti_x_pow(enhance_image, r1, r2, r3)
            r_2 = torch.cat([r1, r2, r3], 1)

            return enhance_image, enhance_image_stage2, r_1, r_2
        else:

            out = self.bottleneck2(enhance_image)
            out = self.conv_to(out)
            r1, r2, r3 = torch.split(torch.nn.functional.tanh(out), 3, dim=1)
            enhance_image = self.muti_x_pow(enhance_image, r1, r2, r3)


            return enhance_image


if __name__ == "__main__":
    from torchsummary import summary
    import torch
    import thop
    DCE_net = enhanceNet()
    DCE_net.eval()
    input_size = 512
    summary(DCE_net, input_size=(3, input_size, input_size), device='cpu')
    inputs = torch.randn(1, 3, input_size, input_size)
    flops, params = thop.profile(DCE_net, inputs=(inputs,))
    gflops = flops / (10 ** 9)
    print("GFLOPs: %.2f" % gflops)


