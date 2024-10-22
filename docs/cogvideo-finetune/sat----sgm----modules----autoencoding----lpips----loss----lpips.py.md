# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\lpips\loss\lpips.py`

```py
"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""  # 引用的模块链接说明

from collections import namedtuple  # 从 collections 模块导入 namedtuple，用于创建可命名的元组

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from torchvision import models  # 从 torchvision 导入模型模块

from ..util import get_ckpt_path  # 从父目录的 util 模块导入获取检查点路径的函数


class LPIPS(nn.Module):  # 定义 LPIPS 类，继承自 nn.Module
    # Learned perceptual metric  # 学习到的感知度量
    def __init__(self, use_dropout=True):  # 初始化方法，接受使用 dropout 的参数
        super().__init__()  # 调用父类的初始化方法
        self.scaling_layer = ScalingLayer()  # 实例化 ScalingLayer
        self.chns = [64, 128, 256, 512, 512]  # 定义 VGG16 的特征通道数量
        self.net = vgg16(pretrained=True, requires_grad=False)  # 加载预训练的 VGG16 网络，不更新参数
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)  # 创建第一个线性层
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)  # 创建第二个线性层
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)  # 创建第三个线性层
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)  # 创建第四个线性层
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)  # 创建第五个线性层
        self.load_from_pretrained()  # 加载预训练的权重
        for param in self.parameters():  # 遍历模型参数
            param.requires_grad = False  # 不更新任何参数的梯度


    def load_from_pretrained(self, name="vgg_lpips"):  # 从预训练模型加载参数的方法
        ckpt = get_ckpt_path(name, "sgm/modules/autoencoding/lpips/loss")  # 获取检查点路径
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)  # 加载权重
        print("loaded pretrained LPIPS loss from {}".format(ckpt))  # 打印加载信息


    @classmethod  # 将该方法定义为类方法
    def from_pretrained(cls, name="vgg_lpips"):  # 通过预训练权重创建模型的类方法
        if name != "vgg_lpips":  # 检查模型名称
            raise NotImplementedError  # 抛出未实现错误
        model = cls()  # 实例化模型
        ckpt = get_ckpt_path(name)  # 获取检查点路径
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)  # 加载权重
        return model  # 返回模型


    def forward(self, input, target):  # 前向传播方法，接受输入和目标
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))  # 对输入和目标应用缩放层
        outs0, outs1 = self.net(in0_input), self.net(in1_input)  # 通过网络计算输出
        feats0, feats1, diffs = {}, {}, {}  # 初始化特征和差异字典
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]  # 收集线性层
        for kk in range(len(self.chns)):  # 遍历通道数量
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])  # 规范化输出特征
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2  # 计算特征差的平方

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]  # 计算每个通道的空间平均
        val = res[0]  # 初始化结果为第一个通道的结果
        for l in range(1, len(self.chns)):  # 遍历其余通道
            val += res[l]  # 累加结果
        return val  # 返回最终结果


class ScalingLayer(nn.Module):  # 定义缩放层类
    def __init__(self):  # 初始化方法
        super(ScalingLayer, self).__init__()  # 调用父类初始化方法
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])  # 注册偏移量缓冲区
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])  # 注册缩放量缓冲区

    def forward(self, inp):  # 前向传播方法
        return (inp - self.shift) / self.scale  # 返回缩放后的输入


class NetLinLayer(nn.Module):  # 定义线性层类
    """A single linear layer which does a 1x1 conv"""  # 单个线性层，执行 1x1 卷积
    # 初始化 NetLinLayer 类的构造函数
        def __init__(self, chn_in, chn_out=1, use_dropout=False):
            # 调用父类构造函数初始化
            super(NetLinLayer, self).__init__()
            # 根据是否使用 dropout 创建层列表
            layers = (
                [
                    nn.Dropout(),  # 添加 dropout 层以防止过拟合
                ]
                if (use_dropout)  # 检查是否需要使用 dropout
                else []  # 如果不需要，返回空列表
            )
            # 在层列表中添加卷积层
            layers += [
                nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),  # 添加卷积层，卷积核大小为1x1
            ]
            # 将层列表封装为顺序模型
            self.model = nn.Sequential(*layers)  # 使用 nn.Sequential 组合所有层
# 定义一个名为 vgg16 的类，继承自 PyTorch 的 nn.Module
class vgg16(torch.nn.Module):
    # 初始化方法，接受是否需要梯度和是否使用预训练模型的参数
    def __init__(self, requires_grad=False, pretrained=True):
        # 调用父类的初始化方法
        super(vgg16, self).__init__()
        # 获取 VGG16 的预训练特征部分
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        # 创建五个序列容器，用于存放不同层的特征提取
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        # 定义切片的数量
        self.N_slices = 5
        # 将前4层特征添加到 slice1
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # 将第4到第8层特征添加到 slice2
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # 将第9到第15层特征添加到 slice3
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # 将第16到第22层特征添加到 slice4
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # 将第23到第29层特征添加到 slice5
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        # 如果不需要计算梯度，则将所有参数的 requires_grad 属性设置为 False
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    # 定义前向传播方法
    def forward(self, X):
        # 通过 slice1 处理输入 X
        h = self.slice1(X)
        # 保存 slice1 的输出
        h_relu1_2 = h
        # 通过 slice2 处理 h
        h = self.slice2(h)
        # 保存 slice2 的输出
        h_relu2_2 = h
        # 通过 slice3 处理 h
        h = self.slice3(h)
        # 保存 slice3 的输出
        h_relu3_3 = h
        # 通过 slice4 处理 h
        h = self.slice4(h)
        # 保存 slice4 的输出
        h_relu4_3 = h
        # 通过 slice5 处理 h
        h = self.slice5(h)
        # 保存 slice5 的输出
        h_relu5_3 = h
        # 创建一个命名元组，包含各个 ReLU 层的输出
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        # 将各层的输出组合成一个命名元组
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        # 返回结果
        return out


# 定义一个归一化张量的函数，接受一个张量和一个小的 epsilon 值
def normalize_tensor(x, eps=1e-10):
    # 计算张量 x 在第1维的 L2 范数
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    # 返回归一化后的张量，避免除以零
    return x / (norm_factor + eps)


# 定义一个空间平均的函数，接受一个张量和一个布尔值 keepdim
def spatial_average(x, keepdim=True):
    # 在空间维度（高度和宽度）上计算均值，保留维度选项
    return x.mean([2, 3], keepdim=keepdim)
```