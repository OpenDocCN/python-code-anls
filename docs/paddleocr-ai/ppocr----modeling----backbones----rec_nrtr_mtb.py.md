# `.\PaddleOCR\ppocr\modeling\backbones\rec_nrtr_mtb.py`

```py
# 导入 paddle.nn 模块
from paddle import nn
# 导入 paddle 模块
import paddle

# 定义 MTB 类，继承自 nn.Layer 类
class MTB(nn.Layer):
    # 初始化函数，接受 cnn_num 和 in_channels 两个参数
    def __init__(self, cnn_num, in_channels):
        # 调用父类的初始化函数
        super(MTB, self).__init__()
        # 初始化 block 层为 nn.Sequential()
        self.block = nn.Sequential()
        # 初始化 out_channels 为 in_channels
        self.out_channels = in_channels
        # 初始化 cnn_num
        self.cnn_num = cnn_num
        # 如果 cnn_num 为 2
        if self.cnn_num == 2:
            # 循环两次
            for i in range(self.cnn_num):
                # 添加卷积层到 block 中
                self.block.add_sublayer(
                    'conv_{}'.format(i),
                    nn.Conv2D(
                        in_channels=in_channels
                        if i == 0 else 32 * (2**(i - 1)),
                        out_channels=32 * (2**i),
                        kernel_size=3,
                        stride=2,
                        padding=1))
                # 添加 ReLU 激活函数到 block 中
                self.block.add_sublayer('relu_{}'.format(i), nn.ReLU())
                # 添加 BatchNorm2D 到 block 中
                self.block.add_sublayer('bn_{}'.format(i),
                                        nn.BatchNorm2D(32 * (2**i)))

    # 前向传播函数，接受 images 参数
    def forward(self, images):
        # 将 images 传入 block 中
        x = self.block(images)
        # 如果 cnn_num 为 2
        if self.cnn_num == 2:
            # 调整 x 的维度顺序
            x = paddle.transpose(x, [0, 3, 2, 1])
            # 获取 x 的形状
            x_shape = paddle.shape(x)
            # 重塑 x 的形状
            x = paddle.reshape(
                x, [x_shape[0], x_shape[1], x_shape[2] * x_shape[3]])
        # 返回 x
        return x
```