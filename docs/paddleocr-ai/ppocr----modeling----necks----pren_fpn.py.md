# `.\PaddleOCR\ppocr\modeling\necks\pren_fpn.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按"原样"分发，不附带任何担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制
"""
# 代码参考自：
# https://github.com/RuijieJ/pren/blob/main/Nets/Aggregation.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle 库
import paddle
from paddle import nn
import paddle.nn.functional as F

# 定义一个名为 PoolAggregate 的类，继承自 nn.Layer
class PoolAggregate(nn.Layer):
    def __init__(self, n_r, d_in, d_middle=None, d_out=None):
        super(PoolAggregate, self).__init__()
        # 如果未指定 d_middle，则设为 d_in
        if not d_middle:
            d_middle = d_in
        # 如果未指定 d_out，则设为 d_in
        if not d_out:
            d_out = d_in

        # 初始化类的属性
        self.d_in = d_in
        self.d_middle = d_middle
        self.d_out = d_out
        self.act = nn.Swish()

        self.n_r = n_r
        # 构建聚合层
        self.aggs = self._build_aggs()

    # 构建聚合层的方法
    def _build_aggs(self):
        aggs = []
        # 循环创建 n_r 个聚合层
        for i in range(self.n_r):
            aggs.append(
                self.add_sublayer(
                    '{}'.format(i),
                    nn.Sequential(
                        ('conv1', nn.Conv2D(
                            self.d_in, self.d_middle, 3, 2, 1, bias_attr=False)
                         ), ('bn1', nn.BatchNorm(self.d_middle)),
                        ('act', self.act), ('conv2', nn.Conv2D(
                            self.d_middle, self.d_out, 3, 2, 1, bias_attr=False
                        )), ('bn2', nn.BatchNorm(self.d_out))))
            )
        return aggs
    # 定义一个前向传播函数，接受输入张量 x
    def forward(self, x):
        # 获取输入张量 x 的 batch 大小
        b = x.shape[0]
        # 初始化一个空列表 outs 用于存储每个聚合函数的输出
        outs = []
        # 遍历每个聚合函数
        for agg in self.aggs:
            # 对输入张量 x 进行聚合操作，得到输出张量 y
            y = agg(x)
            # 对输出张量 y 进行自适应平均池化，将其大小调整为 (batch_size, 1, self.d_out)
            p = F.adaptive_avg_pool2d(y, 1)
            # 将调整后的张量 p 重新形状为 (batch_size, 1, self.d_out)，并添加到 outs 列表中
            outs.append(p.reshape((b, 1, self.d_out)))
        # 将所有聚合函数的输出张量在维度 1 上进行拼接，得到最终的输出张量 out
        out = paddle.concat(outs, 1)
        # 返回最终的输出张量
        return out
# 定义 WeightAggregate 类，用于权重聚合操作
class WeightAggregate(nn.Layer):
    def __init__(self, n_r, d_in, d_middle=None, d_out=None):
        super(WeightAggregate, self).__init__()
        # 如果未指定中间维度，则设置为输入维度
        if not d_middle:
            d_middle = d_in
        # 如果未指定输出维度，则设置为输入维度
        if not d_out:
            d_out = d_in

        self.n_r = n_r
        self.d_out = d_out
        self.act = nn.Swish()

        # 定义卷积网络 conv_n，包括多个卷积层和批归一化层
        self.conv_n = nn.Sequential(
            ('conv1', nn.Conv2D(
                d_in, d_in, 3, 1, 1,
                bias_attr=False)), ('bn1', nn.BatchNorm(d_in)),
            ('act1', self.act), ('conv2', nn.Conv2D(
                d_in, n_r, 1, bias_attr=False)), ('bn2', nn.BatchNorm(n_r)),
            ('act2', nn.Sigmoid()))
        # 定义卷积网络 conv_d，包括多个卷积层和批归一化层
        self.conv_d = nn.Sequential(
            ('conv1', nn.Conv2D(
                d_in, d_middle, 3, 1, 1,
                bias_attr=False)), ('bn1', nn.BatchNorm(d_middle)),
            ('act1', self.act), ('conv2', nn.Conv2D(
                d_middle, d_out, 1,
                bias_attr=False)), ('bn2', nn.BatchNorm(d_out)))

    # 前向传播函数
    def forward(self, x):
        b, _, h, w = x.shape

        # 对输入数据进行权重聚合操作
        hmaps = self.conv_n(x)
        fmaps = self.conv_d(x)
        r = paddle.bmm(
            hmaps.reshape((b, self.n_r, h * w)),
            fmaps.reshape((b, self.d_out, h * w)).transpose((0, 2, 1)))
        return r


# 定义 GCN 类，用于图卷积网络操作
class GCN(nn.Layer):
    def __init__(self, d_in, n_in, d_out=None, n_out=None, dropout=0.1):
        super(GCN, self).__init__()
        # 如果未指定输出维度，则设置为输入维度
        if not d_out:
            d_out = d_in
        # 如果未指定输出节点数，则设置为输入维度
        if not n_out:
            n_out = d_in

        # 定义卷积网络 conv_n，包括一个一维卷积层
        self.conv_n = nn.Conv1D(n_in, n_out, 1)
        # 定义线性层
        self.linear = nn.Linear(d_in, d_out)
        # 定义 dropout 层
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Swish()

    # 前向传播函数
    def forward(self, x):
        # 对输入数据进行一维卷积操作
        x = self.conv_n(x)
        # 对卷积结果进行线性变换和 dropout 操作
        x = self.dropout(self.linear(x))
        return self.act(x)


class PRENFPN(nn.Layer):
    # 初始化函数，定义了模型的结构和参数
    def __init__(self, in_channels, n_r, d_model, max_len, dropout):
        # 调用父类的初始化函数
        super(PRENFPN, self).__init__()
        # 断言输入通道数为3
        assert len(in_channels) == 3, "in_channels' length must be 3."
        c1, c2, c3 = in_channels  # the depths are from big to small
        # 构建特征金字塔网络
        assert d_model % 3 == 0, "{} can't be divided by 3.".format(d_model)
        self.agg_p1 = PoolAggregate(n_r, c1, d_out=d_model // 3)
        self.agg_p2 = PoolAggregate(n_r, c2, d_out=d_model // 3)
        self.agg_p3 = PoolAggregate(n_r, c3, d_out=d_model // 3)

        self.agg_w1 = WeightAggregate(n_r, c1, 4 * c1, d_model // 3)
        self.agg_w2 = WeightAggregate(n_r, c2, 4 * c2, d_model // 3)
        self.agg_w3 = WeightAggregate(n_r, c3, 4 * c3, d_model // 3)

        self.gcn_pool = GCN(d_model, n_r, d_model, max_len, dropout)
        self.gcn_weight = GCN(d_model, n_r, d_model, max_len, dropout)

        self.out_channels = d_model

    # 前向传播函数，定义了数据在模型中的流动过程
    def forward(self, inputs):
        f3, f5, f7 = inputs

        rp1 = self.agg_p1(f3)
        rp2 = self.agg_p2(f5)
        rp3 = self.agg_p3(f7)
        rp = paddle.concat([rp1, rp2, rp3], 2)  # [b,nr,d]

        rw1 = self.agg_w1(f3)
        rw2 = self.agg_w2(f5)
        rw3 = self.agg_w3(f7)
        rw = paddle.concat([rw1, rw2, rw3], 2)  # [b,nr,d]

        y1 = self.gcn_pool(rp)
        y2 = self.gcn_weight(rw)
        y = 0.5 * (y1 + y2)
        return y  # [b,max_len,d]
```