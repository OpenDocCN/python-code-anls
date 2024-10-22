# `.\cogview3-finetune\sat\sgm\modules\autoencoding\lpips\loss\lpips.py`

```
# 从 https://github.com/richzhang/PerceptualSimilarity/tree/master/models 中剥离的版本
"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

# 从 collections 模块导入 namedtuple
from collections import namedtuple

# 导入 PyTorch 和神经网络模块
import torch
import torch.nn as nn
# 从 torchvision 导入模型
from torchvision import models

# 从上层目录的 util 模块导入获取检查点路径的函数
from ..util import get_ckpt_path


# 定义 LPIPS 类，继承自 nn.Module
class LPIPS(nn.Module):
    # 学习的感知度量
    def __init__(self, use_dropout=True):
        # 调用父类构造函数
        super().__init__()
        # 初始化缩放层
        self.scaling_layer = ScalingLayer()
        # 定义特征通道数量，针对 VGG16 特征
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        # 加载预训练的 VGG16 模型，不计算其梯度
        self.net = vgg16(pretrained=True, requires_grad=False)
        # 为每个通道初始化线性层
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        # 从预训练模型中加载权重
        self.load_from_pretrained()
        # 禁用所有参数的梯度更新
        for param in self.parameters():
            param.requires_grad = False

    # 从预训练模型加载权重的方法
    def load_from_pretrained(self, name="vgg_lpips"):
        # 获取检查点文件的路径
        ckpt = get_ckpt_path(name, "sgm/modules/autoencoding/lpips/loss")
        # 加载权重到当前模型中，严格匹配状态字典
        self.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        # 打印加载的模型路径
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    # 类方法，用于从预训练模型创建实例
    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        # 检查模型名称是否有效
        if name != "vgg_lpips":
            raise NotImplementedError
        # 创建当前类的实例
        model = cls()
        # 获取检查点路径
        ckpt = get_ckpt_path(name)
        # 加载权重到模型中
        model.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        # 返回模型实例
        return model

    # 前向传播方法
    def forward(self, input, target):
        # 对输入和目标进行缩放处理
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        # 通过网络提取特征
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        # 初始化特征和差异字典
        feats0, feats1, diffs = {}, {}, {}
        # 收集线性层
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        # 遍历特征通道
        for kk in range(len(self.chns)):
            # 标准化特征并计算差异
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(
                outs1[kk]
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        # 计算每个通道的结果
        res = [
            spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
            for kk in range(len(self.chns))
        ]
        # 初始化最终值
        val = res[0]
        # 累加结果
        for l in range(1, len(self.chns)):
            val += res[l]
        # 返回最终的值
        return val


# 定义缩放层类，继承自 nn.Module
class ScalingLayer(nn.Module):
    # 构造函数
    def __init__(self):
        # 调用父类构造函数
        super(ScalingLayer, self).__init__()
        # 注册偏移量的缓冲区
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        # 注册缩放因子的缓冲区
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    # 前向传播方法
    def forward(self, inp):
        # 进行缩放操作
        return (inp - self.shift) / self.scale


# 定义线性层类，继承自 nn.Module
class NetLinLayer(nn.Module):
    """一个单独的线性层，执行 1x1 卷积"""
    # 初始化网络层，接收输入通道数、输出通道数和是否使用 dropout
        def __init__(self, chn_in, chn_out=1, use_dropout=False):
            # 调用父类的初始化方法
            super(NetLinLayer, self).__init__()
            # 根据是否使用 dropout 创建层列表
            layers = (
                [
                    nn.Dropout(),  # 添加 dropout 层
                ]
                if (use_dropout)  # 如果使用 dropout
                else []  # 否则为空列表
            )
            # 添加卷积层到层列表，输入通道为 chn_in，输出通道为 chn_out，卷积核大小为 1
            layers += [
                nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
            ]
            # 将层列表包装成一个顺序容器，便于按顺序调用各层
            self.model = nn.Sequential(*layers)
# 定义一个 VGG16 类，继承自 PyTorch 的 Module 类
class vgg16(torch.nn.Module):
    # 初始化函数，接受是否需要梯度和是否使用预训练模型的参数
    def __init__(self, requires_grad=False, pretrained=True):
        # 调用父类的初始化函数
        super(vgg16, self).__init__()
        # 获取预训练的 VGG16 特征提取层
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        # 定义五个序列层
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        # 设置切片的数量为 5
        self.N_slices = 5
        # 将前4层添加到 slice1 中
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # 将第 4 到 8 层添加到 slice2 中
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # 将第 9 到 15 层添加到 slice3 中
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # 将第 16 到 22 层添加到 slice4 中
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # 将第 23 到 29 层添加到 slice5 中
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        # 如果不需要梯度，则冻结所有参数
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    # 定义前向传播函数
    def forward(self, X):
        # 将输入通过 slice1 层
        h = self.slice1(X)
        # 保存第一层的输出
        h_relu1_2 = h
        # 将输出通过 slice2 层
        h = self.slice2(h)
        # 保存第二层的输出
        h_relu2_2 = h
        # 将输出通过 slice3 层
        h = self.slice3(h)
        # 保存第三层的输出
        h_relu3_3 = h
        # 将输出通过 slice4 层
        h = self.slice4(h)
        # 保存第四层的输出
        h_relu4_3 = h
        # 将输出通过 slice5 层
        h = self.slice5(h)
        # 保存第五层的输出
        h_relu5_3 = h
        # 创建一个命名元组来存储不同层的输出
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        # 将各层输出组合成一个元组
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        # 返回元组
        return out


# 定义一个函数，用于规范化张量
def normalize_tensor(x, eps=1e-10):
    # 计算张量的 L2 范数
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    # 返回规范化后的张量，避免除以零
    return x / (norm_factor + eps)


# 定义一个空间平均函数
def spatial_average(x, keepdim=True):
    # 在高和宽维度上计算平均值
    return x.mean([2, 3], keepdim=keepdim)
```