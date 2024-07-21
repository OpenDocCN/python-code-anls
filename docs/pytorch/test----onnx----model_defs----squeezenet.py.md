# `.\pytorch\test\onnx\model_defs\squeezenet.py`

```
# 导入 PyTorch 库
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入初始化模块
import torch.nn.init as init

# 定义 Fire 模块，继承自 nn.Module
class Fire(nn.Module):
    # 初始化函数，接收输入平面数、压缩平面数、1x1扩展平面数、3x3扩展平面数
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super().__init__()
        # 设置输入平面数
        self.inplanes = inplanes
        # 压缩层：1x1卷积
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        # 压缩层激活函数：ReLU
        self.squeeze_activation = nn.ReLU(inplace=True)
        # 1x1扩展层：1x1卷积
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        # 1x1扩展层激活函数：ReLU
        self.expand1x1_activation = nn.ReLU(inplace=True)
        # 3x3扩展层：3x3卷积，padding为1
        self.expand3x3 = nn.Conv2d(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1
        )
        # 3x3扩展层激活函数：ReLU
        self.expand3x3_activation = nn.ReLU(inplace=True)

    # 前向传播函数
    def forward(self, x):
        # 将输入经过压缩层和激活函数处理
        x = self.squeeze_activation(self.squeeze(x))
        # 对压缩后的结果分别进行1x1和3x3扩展，然后进行ReLU激活，并在通道维度上连接起来
        return torch.cat(
            [
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x)),
            ],
            1,
        )

# 定义 SqueezeNet 模型，继承自 nn.Module
class SqueezeNet(nn.Module):
    # 定义 SqueezeNet 的初始化方法，接受版本号、类别数和 ceil_mode 作为参数
    def __init__(self, version=1.0, num_classes=1000, ceil_mode=False):
        # 调用父类的初始化方法
        super().__init__()
        
        # 检查版本号是否合法，只支持版本号 1.0 和 1.1
        if version not in [1.0, 1.1]:
            raise ValueError(
                f"Unsupported SqueezeNet version {version}: 1.0 or 1.1 expected"
            )
        
        # 设置网络的类别数
        self.num_classes = num_classes
        
        # 根据不同的版本号设置网络的特征提取部分
        if version == 1.0:
            # 版本 1.0 的特征提取部分包括多个卷积层、ReLU激活函数和 Fire 模块
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),  # 第一个卷积层
                nn.ReLU(inplace=True),                     # 激活函数 ReLU
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode),  # 最大池化层
                Fire(96, 16, 64, 64),                      # Fire 模块
                Fire(128, 16, 64, 64),                     # Fire 模块
                Fire(128, 32, 128, 128),                   # Fire 模块
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode),  # 最大池化层
                Fire(256, 32, 128, 128),                   # Fire 模块
                Fire(256, 48, 192, 192),                   # Fire 模块
                Fire(384, 48, 192, 192),                   # Fire 模块
                Fire(384, 64, 256, 256),                   # Fire 模块
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode),  # 最大池化层
                Fire(512, 64, 256, 256),                   # Fire 模块
            )
        else:
            # 版本 1.1 的特征提取部分包括不同的卷积层和 Fire 模块组合
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),  # 第一个卷积层
                nn.ReLU(inplace=True),                     # 激活函数 ReLU
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode),  # 最大池化层
                Fire(64, 16, 64, 64),                      # Fire 模块
                Fire(128, 16, 64, 64),                     # Fire 模块
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode),  # 最大池化层
                Fire(128, 32, 128, 128),                   # Fire 模块
                Fire(256, 32, 128, 128),                   # Fire 模块
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode),  # 最大池化层
                Fire(256, 48, 192, 192),                   # Fire 模块
                Fire(384, 48, 192, 192),                   # Fire 模块
                Fire(384, 64, 256, 256),                   # Fire 模块
                Fire(512, 64, 256, 256),                   # Fire 模块
            )
        
        # 最终的分类器部分包括 Dropout、最后的卷积层、ReLU激活函数和平均池化层
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),        # Dropout 层
            final_conv,               # 最后的卷积层
            nn.ReLU(inplace=True),    # 激活函数 ReLU
            nn.AvgPool2d(13)          # 平均池化层
        )
        
        # 对网络的所有模块进行初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight.data, mean=0.0, std=0.01)  # 最后的卷积层使用正态分布初始化
                else:
                    init.kaiming_uniform_(m.weight.data)  # 其他卷积层使用 Kaiming 初始化
                if m.bias is not None:
                    m.bias.data.zero_()  # 如果有偏置项，将其初始化为零
    
    # 定义 SqueezeNet 的前向传播方法
    def forward(self, x):
        x = self.features(x)   # 前向传播中的特征提取部分
        x = self.classifier(x)  # 前向传播中的分类器部分
        return x.view(x.size(0), self.num_classes)  # 将输出 reshape 成 (batch_size, num_classes)
```