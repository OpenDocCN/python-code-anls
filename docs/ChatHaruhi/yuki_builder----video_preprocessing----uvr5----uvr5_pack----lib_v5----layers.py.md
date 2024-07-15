# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\lib_v5\layers.py`

```py
class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPPModule, self).__init__()
        # 创建一个空的列表 rates，这里默认是[6, 12, 18]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # 使用 rates 列表中的每个元素创建 ASPP 分支
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[0], dilation=rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 对输入 x 分别进行 ASPP 分支的计算，并进行通道拼接
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        # 将四个分支的结果在通道维度上拼接起来
        x = torch.cat((x1, x2, x3, x4), dim=1)
        # 最后对拼接后的结果进行一次卷积操作
        x = self.conv5(x)
        return x
    def __init__(self, nin, nout, dilations=(4, 8, 16), activ=nn.ReLU):
        super(ASPPModule, self).__init__()
        
        # 第一个卷积模块，使用全局平均池化后进行 1x1 卷积
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # 自适应平均池化层，输出尺寸为 (1, w)，其中 w 是输入的宽度
            Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ),  # 1x1 卷积层，不改变特征图的大小
        )
        
        # 第二个卷积模块，简单的 1x1 卷积
        self.conv2 = Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ)
        
        # 第三个卷积模块，使用可分离卷积，带有扩张率
        self.conv3 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[0], dilations[0], activ=activ
        )
        
        # 第四个卷积模块，使用可分离卷积，带有更大的扩张率
        self.conv4 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[1], dilations[1], activ=activ
        )
        
        # 第五个卷积模块，使用可分离卷积，带有最大的扩张率
        self.conv5 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[2], dilations[2], activ=activ
        )
        
        # 瓶颈层，将所有卷积的输出进行拼接后，再进行 1x1 卷积和 dropout 操作
        self.bottleneck = nn.Sequential(
            Conv2DBNActiv(nin * 5, nout, 1, 1, 0, activ=activ),  # 1x1 卷积层，将五个特征图拼接起来
            nn.Dropout2d(0.1)  # dropout 操作，防止过拟合
        )

    def forward(self, x):
        _, _, h, w = x.size()
        
        # 提取特征，进行上采样使得特征图与输入大小相同
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True
        )
        
        # 直接通过第二个卷积模块处理输入特征
        feat2 = self.conv2(x)
        
        # 使用第三个卷积模块处理输入特征
        feat3 = self.conv3(x)
        
        # 使用第四个卷积模块处理输入特征
        feat4 = self.conv4(x)
        
        # 使用第五个卷积模块处理输入特征
        feat5 = self.conv5(x)
        
        # 将所有特征图拼接在一起
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        
        # 使用瓶颈层处理拼接后的特征图
        bottle = self.bottleneck(out)
        
        # 返回处理后的特征图
        return bottle
```