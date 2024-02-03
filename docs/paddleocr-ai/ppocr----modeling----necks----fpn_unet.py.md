# `.\PaddleOCR\ppocr\modeling\necks\fpn_unet.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按"原样"分发，不附带任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 代码参考自：
# https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/necks/fpn_unet.py

# 导入 PaddlePaddle 深度学习框架
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# 定义一个上采样模块类
class UpBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 断言输入的通道数和输出的通道数为整数
        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)

        # 定义一个 1x1 卷积层，用于调整输入通道数
        self.conv1x1 = nn.Conv2D(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 定义一个 3x3 卷积层，用于特征提取
        self.conv3x3 = nn.Conv2D(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 定义一个反卷积层，用于上采样
        self.deconv = nn.Conv2DTranspose(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    # 前向传播函数
    def forward(self, x):
        # 对输入进行 1x1 卷积和激活函数处理
        x = F.relu(self.conv1x1(x))
        # 对处理后的特征进行 3x3 卷积和激活函数处理
        x = F.relu(self.conv3x3(x))
        # 对处理后的特征进行反卷积操作
        x = self.deconv(x)
        return x

# 定义 FPN_UNet 类
class FPN_UNet(nn.Layer):
    # 初始化函数，定义 UNet 模型的结构
    def __init__(self, in_channels, out_channels):
        # 调用父类的初始化函数
        super().__init__()

        # 断言输入通道数为4
        assert len(in_channels) == 4
        # 断言输出通道数为整数
        assert isinstance(out_channels, int)
        self.out_channels = out_channels

        # 计算每个块的输出通道数
        blocks_out_channels = [out_channels] + [
            min(out_channels * 2**i, 256) for i in range(4)
        ]
        # 计算每个块的输入通道数
        blocks_in_channels = [blocks_out_channels[1]] + [
            in_channels[i] + blocks_out_channels[i + 2] for i in range(3)
        ] + [in_channels[3]]

        # 定义上采样层
        self.up4 = nn.Conv2DTranspose(
            blocks_in_channels[4],
            blocks_out_channels[4],
            kernel_size=4,
            stride=2,
            padding=1)
        self.up_block3 = UpBlock(blocks_in_channels[3], blocks_out_channels[3])
        self.up_block2 = UpBlock(blocks_in_channels[2], blocks_out_channels[2])
        self.up_block1 = UpBlock(blocks_in_channels[1], blocks_out_channels[1])
        self.up_block0 = UpBlock(blocks_in_channels[0], blocks_out_channels[0])

    # 前向传播函数
    def forward(self, x):
        """
        Args:
            x (list[Tensor] | tuple[Tensor]): A list of four tensors of shape
                :math:`(N, C_i, H_i, W_i)`, representing C2, C3, C4, C5
                features respectively. :math:`C_i` should matches the number in
                ``in_channels``.

        Returns:
            Tensor: Shape :math:`(N, C, H, W)` where :math:`H=4H_0` and
            :math:`W=4W_0`.
        """
        # 将输入特征分别赋值给 c2, c3, c4, c5
        c2, c3, c4, c5 = x

        # 对 c5 进行上采样并激活
        x = F.relu(self.up4(c5))

        # 将上采样结果与 c4 进行拼接，并通过上采样块3处理
        x = paddle.concat([x, c4], axis=1)
        x = F.relu(self.up_block3(x))

        # 将处理后的结果与 c3 进行拼接，并通过上采样块2处理
        x = paddle.concat([x, c3], axis=1)
        x = F.relu(self.up_block2(x))

        # 将处理后的结果与 c2 进行拼接，并通过上采样块1处理
        x = paddle.concat([x, c2], axis=1)
        x = F.relu(self.up_block1(x))

        # 最后通过上采样块0处理得到最终输出
        x = self.up_block0(x)
        return x
```