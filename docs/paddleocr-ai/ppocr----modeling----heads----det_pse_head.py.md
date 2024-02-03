# `.\PaddleOCR\ppocr\modeling\heads\det_pse_head.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息
"""
# 引用的代码来源于：
# https://github.com/whai362/PSENet/blob/python3/models/head/psenet_head.py

from paddle import nn

# 定义 PSEHead 类，继承自 nn.Layer
class PSEHead(nn.Layer):
    # 初始化方法，接受输入通道数、隐藏维度、输出通道数和其他参数
    def __init__(self, in_channels, hidden_dim=256, out_channels=7, **kwargs):
        super(PSEHead, self).__init__()
        # 第一个卷积层，输入通道数为 in_channels，输出通道数为 hidden_dim，卷积核大小为 3x3，步长为 1，填充为 1
        self.conv1 = nn.Conv2D(
            in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        # 第一个批归一化层，输入通道数为 hidden_dim
        self.bn1 = nn.BatchNorm2D(hidden_dim)
        # 第一个激活函数层，使用 ReLU 激活函数
        self.relu1 = nn.ReLU()

        # 第二个卷积层，输入通道数为 hidden_dim，输出通道数为 out_channels，卷积核大小为 1x1，步长为 1，填充为 0
        self.conv2 = nn.Conv2D(
            hidden_dim, out_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播方法，接受输入 x 和其他参数
    def forward(self, x, **kwargs):
        # 第一层卷积操作
        out = self.conv1(x)
        # 第一层批归一化和激活函数操作
        out = self.relu1(self.bn1(out))
        # 第二层卷积操作
        out = self.conv2(out)
        # 返回结果字典，包含 'maps' 键和卷积结果值
        return {'maps': out}
```