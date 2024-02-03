# `.\PaddleOCR\ppocr\modeling\heads\det_fce_head.py`

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
# 均按“原样”分发，不附带任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的详细信息
"""
# 代码参考自：
# https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/dense_heads/fce_head.py

# 导入所需的库和模块
from paddle import nn
from paddle import ParamAttr
import paddle.nn.functional as F
from paddle.nn.initializer import Normal
import paddle
from functools import partial

# 定义一个函数，将函数 func 应用于多个参数
def multi_apply(func, *args, **kwargs):
    # 如果有关键字参数 kwargs，则使用 partial 函数将其绑定到 func 上
    pfunc = partial(func, **kwargs) if kwargs else func
    # 使用 map 函数将 pfunc 应用于 args 的每个元素
    map_results = map(pfunc, *args)
    # 将结果转换为元组并返回
    return tuple(map(list, zip(*map_results)))

# 定义一个类，实现 FCENet 头部
class FCEHead(nn.Layer):
    """The class for implementing FCENet head.
    FCENet(CVPR2021): Fourier Contour Embedding for Arbitrary-shaped Text
    Detection.

    [https://arxiv.org/abs/2104.10442]

    Args:
        in_channels (int): The number of input channels.
        scales (list[int]) : The scale of each layer.
        fourier_degree (int) : The maximum Fourier transform degree k.
    """
    # 初始化函数，设置输入通道数和傅里叶级数，默认为5
    def __init__(self, in_channels, fourier_degree=5):
        # 调用父类的初始化函数
        super().__init__()
        # 断言输入通道数为整数类型
        assert isinstance(in_channels, int)

        # 下采样比例为1.0
        self.downsample_ratio = 1.0
        # 输入通道数
        self.in_channels = in_channels
        # 傅里叶级数
        self.fourier_degree = fourier_degree
        # 分类输出通道数为4
        self.out_channels_cls = 4
        # 回归输出通道数为(2 * 傅里叶级数 + 1) * 2
        self.out_channels_reg = (2 * self.fourier_degree + 1) * 2

        # 分类输出卷积层
        self.out_conv_cls = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels_cls,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            weight_attr=ParamAttr(
                name='cls_weights',
                initializer=Normal(
                    mean=0., std=0.01)),
            bias_attr=True)
        # 回归输出卷积层
        self.out_conv_reg = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels_reg,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            weight_attr=ParamAttr(
                name='reg_weights',
                initializer=Normal(
                    mean=0., std=0.01)),
            bias_attr=True)

    # 前向传播函数
    def forward(self, feats, targets=None):
        # 对特征进行单独前向传播
        cls_res, reg_res = multi_apply(self.forward_single, feats)
        # 获取分类结果和回归结果的数量
        level_num = len(cls_res)
        outs = {}
        # 如果不是训练阶段
        if not self.training:
            # 对每个级别的分类结果进行softmax处理
            for i in range(level_num):
                tr_pred = F.softmax(cls_res[i][:, 0:2, :, :], axis=1)
                tcl_pred = F.softmax(cls_res[i][:, 2:, :, :], axis=1)
                # 将分类结果、回归结果拼接在一起
                outs['level_{}'.format(i)] = paddle.concat(
                    [tr_pred, tcl_pred, reg_res[i]], axis=1)
        else:
            # 如果是训练阶段，将分类结果和回归结果组合在一起
            preds = [[cls_res[i], reg_res[i]] for i in range(level_num)]
            outs['levels'] = preds
        return outs

    # 单个特征的前向传播函数
    def forward_single(self, x):
        # 分类预测
        cls_predict = self.out_conv_cls(x)
        # 回归预测
        reg_predict = self.out_conv_reg(x)
        return cls_predict, reg_predict
```