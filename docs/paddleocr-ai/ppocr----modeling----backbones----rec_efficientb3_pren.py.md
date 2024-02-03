# `.\PaddleOCR\ppocr\modeling\backbones\rec_efficientb3_pren.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息
"""
代码参考自:
https://github.com/RuijieJ/pren/blob/main/Nets/EfficientNet.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import re
import collections
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ['EfficientNetb3']

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'num_classes',
    'width_coefficient', 'depth_coefficient', 'depth_divisor', 'min_depth',
    'drop_connect_rate', 'image_size'
])

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'
])

# 定义一个 BlockDecoder 类
class BlockDecoder:
    @staticmethod
    # 解码块字符串，将字符串解析为块参数对象
    def _decode_block_string(block_string):
        # 断言块字符串是字符串类型
        assert isinstance(block_string, str)

        # 以'_'为分隔符将块字符串分割成操作列表
        ops = block_string.split('_')
        # 创建空字典用于存储操作和对应的数值
        options = {}
        # 遍历操作列表
        for op in ops:
            # 使用正则表达式将操作分割成键值对
            splits = re.split(r'(\d.*)', op)
            # 如果分割后的列表长度大于等于2
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # 断言操作字典中包含's'键且值长度为1，或者值长度为2且两个值相等
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        # 返回块参数对象，包括核大小、重复次数、输入和输出滤波器数量、扩展比率、是否跳跃连接、SE比率和步长
        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    # 静态方法，解码字符串列表为块参数对象列表
    @staticmethod
    def decode(string_list):
        # 断言输入参数是列表类型
        assert isinstance(string_list, list)
        # 创建空列表用于存储块参数对象
        blocks_args = []
        # 遍历字符串列表
        for block_string in string_list:
            # 调用内部方法解码块字符串并添加到块参数对象列表中
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        # 返回块参数对象列表
        return blocks_args
# 定义一个 efficientnet 函数，用于创建 EfficientNet 模型
def efficientnet(width_coefficient=None,
                 depth_coefficient=None,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 image_size=None,
                 num_classes=1000):
    # 定义一组块参数，描述了每个块的结构
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    # 解码块参数，转换为可用的块参数
    blocks_args = BlockDecoder.decode(blocks_args)

    # 定义全局参数，包括批量归一化的动量、批量归一化的 epsilon、dropout 率等
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size, )
    # 返回块参数和全局参数
    return blocks_args, global_params


# 定义一个 EffUtils 类，包含一些工具方法
class EffUtils:
    @staticmethod
    def round_filters(filters, global_params):
        """ Calculate and round number of filters based on depth multiplier. """
        # 根据深度乘数计算并四舍五入滤波器的数量
        multiplier = global_params.width_coefficient
        if not multiplier:
            return filters
        divisor = global_params.depth_divisor
        min_depth = global_params.min_depth
        filters *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(min_depth,
                          int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    @staticmethod
    def round_repeats(repeats, global_params):
        """ Round number of filters based on depth multiplier. """
        # 根据深度乘数四舍五入重复次数
        multiplier = global_params.depth_coefficient
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))


# 定义一个 MbConvBlock 类，用于实现 MbConv 块
    # 初始化 MbConvBlock 类，传入块参数
    def __init__(self, block_args):
        # 调用父类的初始化方法
        super(MbConvBlock, self).__init__()
        # 保存块参数
        self._block_args = block_args
        # 判断是否需要 Squeeze-and-Excitation 操作
        self.has_se = (self._block_args.se_ratio is not None) and \
            (0 < self._block_args.se_ratio <= 1)
        # 判断是否有跳跃连接
        self.id_skip = block_args.id_skip

        # 扩展阶段
        self.inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            # 创建扩展卷积层
            self._expand_conv = nn.Conv2D(self.inp, oup, 1, bias_attr=False)
            self._bn0 = nn.BatchNorm(oup)

        # 深度卷积阶段
        k = self._block_args.kernel_size
        s = self._block_args.stride
        if isinstance(s, list):
            s = s[0]
        # 创建深度卷积层
        self._depthwise_conv = nn.Conv2D(
            oup,
            oup,
            groups=oup,
            kernel_size=k,
            stride=s,
            padding='same',
            bias_attr=False)
        self._bn1 = nn.BatchNorm(oup)

        # 如果需要，添加 Squeeze-and-Excitation 层
        if self.has_se:
            num_squeezed_channels = max(1,
                                        int(self._block_args.input_filters *
                                            self._block_args.se_ratio))
            self._se_reduce = nn.Conv2D(oup, num_squeezed_channels, 1)
            self._se_expand = nn.Conv2D(num_squeezed_channels, oup, 1)

        # 输出阶段和一些实用类
        self.final_oup = self._block_args.output_filters
        # 创建投影卷积层
        self._project_conv = nn.Conv2D(oup, self.final_oup, 1, bias_attr=False)
        self._bn2 = nn.BatchNorm(self.final_oup)
        self._swish = nn.Swish()
    # 在训练模式下，根据概率 p 进行随机丢弃连接，否则直接返回输入
    def _drop_connect(self, inputs, p, training):
        if not training:
            return inputs
        batch_size = inputs.shape[0]
        keep_prob = 1 - p
        random_tensor = keep_prob
        random_tensor += paddle.rand([batch_size, 1, 1, 1], dtype=inputs.dtype)
        random_tensor = paddle.to_tensor(random_tensor, place=inputs.place)
        binary_tensor = paddle.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output

    # 前向传播函数
    def forward(self, inputs, drop_connect_rate=None):
        # expansion and depthwise conv
        x = inputs
        # 如果扩展比例不为1，则进行扩展卷积操作
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # squeeze and excitation
        # 如果存在 SE 模块，则进行 squeeze 和 excitation 操作
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(
                self._swish(self._se_reduce(x_squeezed)))
            x = F.sigmoid(x_squeezed) * x
        x = self._bn2(self._project_conv(x))

        # skip conntection and drop connect
        # 如果启用了跳跃连接，并且步长为1且输入等于最终输出
        if self.id_skip and self._block_args.stride == 1 and \
            self.inp == self.final_oup:
            # 如果设置了丢弃连接率，则调用 _drop_connect 函数
            if drop_connect_rate:
                x = self._drop_connect(
                    x, p=drop_connect_rate, training=self.training)
            # 将输入和 x 相加作为最终输出
            x = x + inputs
        return x
# 定义 EfficientNetb3_PREN 类，继承自 nn.Layer 类
class EfficientNetb3_PREN(nn.Layer):
    # 定义前向传播函数
    def forward(self, inputs):
        # 初始化输出列表
        outs = []
        # 对输入数据进行初始处理：卷积、批归一化、激活函数
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        # 遍历网络中的每个块
        for idx, block in enumerate(self._blocks):
            # 计算当前块的 drop connect rate
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            # 将数据传入当前块进行处理
            x = block(x, drop_connect_rate=drop_connect_rate)
            # 如果当前块的索引在关注的块索引列表中，则将处理后的数据添加到输出列表中
            if idx in self._concerned_block_idxes:
                outs.append(x)
        # 返回处理后的输出列表
        return outs
```