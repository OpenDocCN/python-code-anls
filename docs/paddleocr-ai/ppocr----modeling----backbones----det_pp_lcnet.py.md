# `.\PaddleOCR\ppocr\modeling\backbones\det_pp_lcnet.py`

```
# 版权声明和许可信息
# 版权所有 (c) 2021 PaddlePaddle 作者。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。

# 导入必要的库
from __future__ import absolute_import, division, print_function
import os
import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn import AdaptiveAvgPool2D, BatchNorm, Conv2D, Dropout, Linear
from paddle.regularizer import L2Decay
from paddle.nn.initializer import KaimingNormal
from paddle.utils.download import get_path_from_url

# 预训练模型的下载链接
MODEL_URLS = {
    "PPLCNet_x0.25":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_25_pretrained.pdparams",
    "PPLCNet_x0.35":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_35_pretrained.pdparams",
    "PPLCNet_x0.5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_5_pretrained.pdparams",
    "PPLCNet_x0.75":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_75_pretrained.pdparams",
    "PPLCNet_x1.0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_0_pretrained.pdparams",
    "PPLCNet_x1.5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_5_pretrained.pdparams",
    "PPLCNet_x2.0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_0_pretrained.pdparams",
    "PPLCNet_x2.5":  # 未完整的链接，需要继续添加
    # 定义一个字符串变量，存储 PaddlePaddle 模型参数文件的下载链接
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_5_pretrained.pdparams"
# 定义模型各个阶段的模式，每个模式对应一个列表，包含不同阶段的名称
MODEL_STAGES_PATTERN = {
    "PPLCNet": ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]
}

# 将所有模型 URL 的键转换为列表
__all__ = list(MODEL_URLS.keys())

# 网络配置，每个阶段包含一个深度块列表，每个深度块由 k, in_c, out_c, s, use_se 组成
# k: 卷积核大小
# in_c: 深度块中的输入通道数
# out_c: 深度块中的输出通道数
# s: 深度块中的步长
# use_se: 是否使用 SE 块
NET_CONFIG = {
    "blocks2":
    # k, in_c, out_c, s, use_se
    [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5":
    [[3, 128, 256, 2, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False],
     [5, 256, 256, 1, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}

# 定义一个函数，用于将输入值调整为可被除数整除的值
def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# 定义一个卷积层和批归一化层的类
class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 num_groups=1):
        super().__init__()

        # 定义卷积层
        self.conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)

        # 定义批归一化层
        self.bn = BatchNorm(
            num_filters,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.hardswish = nn.Hardswish()
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 使用卷积层对输入 x 进行卷积操作
        x = self.conv(x)
        # 对卷积结果进行批量归一化处理
        x = self.bn(x)
        # 对归一化后的结果进行激活函数处理
        x = self.hardswish(x)
        # 返回处理后的结果
        return x
# 定义深度可分离卷积层类
class DepthwiseSeparable(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 dw_size=3,
                 use_se=False):
        super().__init__()
        self.use_se = use_se
        # 深度可分离卷积层，包含深度卷积和逐点卷积
        self.dw_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels)
        # 如果使用 SE 模块
        if use_se:
            self.se = SEModule(num_channels)
        # 逐点卷积层
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1)

    # 前向传播函数
    def forward(self, x):
        # 深度卷积
        x = self.dw_conv(x)
        # 如果使用 SE 模块
        if self.use_se:
            x = self.se(x)
        # 逐点卷积
        x = self.pw_conv(x)
        return x


# 定义 SE 模块类
class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        # 平均池化层
        self.avg_pool = AdaptiveAvgPool2D(1)
        # 第一个卷积层
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        # ReLU 激活函数
        self.relu = nn.ReLU()
        # 第二个卷积层
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        # 硬 Sigmoid 激活函数
        self.hardsigmoid = nn.Hardsigmoid()

    # 前向传播函数
    def forward(self, x):
        # 保存输入
        identity = x
        # 平均池化
        x = self.avg_pool(x)
        # 第一个卷积层
        x = self.conv1(x)
        # ReLU 激活
        x = self.relu(x)
        # 第二个卷积层
        x = self.conv2(x)
        # 硬 Sigmoid 激活
        x = self.hardsigmoid(x)
        # 乘法操作
        x = paddle.multiply(x=identity, y=x)
        return x


class PPLCNet(nn.Layer):
    # 前向传播函数，接收输入 x，经过卷积层和多个残差块后返回多个输出
    def forward(self, x):
        # 用于存储每个阶段的输出
        outs = []
        # 经过第一个卷积层
        x = self.conv1(x)
        # 经过第二个残差块
        x = self.blocks2(x)
        # 经过第三个残差块
        x = self.blocks3(x)
        # 将第三个残差块的输出添加到输出列表中
        outs.append(x)
        # 经过第四个残差块
        x = self.blocks4(x)
        # 将第四个残差块的输出添加到输出列表中
        outs.append(x)
        # 经过第五个残差块
        x = self.blocks5(x)
        # 将第五个残差块的输出添加到输出列表中
        outs.append(x)
        # 经过第六个残差块
        x = self.blocks6(x)
        # 将第六个残差块的输出添加到输出列表中
        outs.append(x)
        # 返回所有阶段的输出
        return outs

    # 加载预训练模型参数的函数，根据预训练模型的 URL 加载参数
    def _load_pretrained(self, pretrained_url, use_ssld=False):
        # 如果使用 SS-LD 模型，则替换预训练模型 URL 中的关键字
        if use_ssld:
            pretrained_url = pretrained_url.replace("_pretrained",
                                                    "_ssld_pretrained")
        # 打印预训练模型的 URL
        print(pretrained_url)
        # 获取本地存储路径
        local_weight_path = get_path_from_url(
            pretrained_url, os.path.expanduser("~/.paddleclas/weights"))
        # 加载本地存储的参数字典
        param_state_dict = paddle.load(local_weight_path)
        # 将加载的参数字典设置到当前模型中
        self.set_dict(param_state_dict)
        return
```