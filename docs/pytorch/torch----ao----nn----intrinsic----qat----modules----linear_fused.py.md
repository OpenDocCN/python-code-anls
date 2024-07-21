# `.\pytorch\torch\ao\nn\intrinsic\qat\modules\linear_fused.py`

```py
# mypy: allow-untyped-defs
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入神经网络模块
import torch.ao.nn.intrinsic as nni  # 导入 AO 神经网络模块
import torch.nn.functional as F  # 导入神经网络函数模块
from torch.nn import init  # 导入初始化函数
from torch.nn.parameter import Parameter  # 导入参数类
from torch.nn.utils.fusion import fuse_linear_bn_weights  # 导入融合函数

__all__ = [
    "LinearBn1d",
]

class LinearBn1d(nn.modules.linear.Linear, nni._FusedModule):
    r"""
    A LinearBn1d module is a module fused from Linear and BatchNorm1d, attached
    with FakeQuantize modules for weight, used in quantization aware training.

    We combined the interface of :class:`torch.nn.Linear` and
    :class:torch.nn.BatchNorm1d`.

    Similar to :class:`torch.nn.Linear`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """
    def __init__(self,
                 # Linear args
                 in_features, out_features, bias=True,
                 # BatchNorm1d args
                 # num_features: out_features
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        # 初始化 LinearBn1d 模块
        nn.modules.linear.Linear.__init__(self, in_features, out_features, bias)
        # 断言确保 qconfig 参数必须被提供用于量化感知训练模块
        assert qconfig, 'qconfig must be provided for QAT module'
        # 设置量化配置
        self.qconfig = qconfig
        # 在训练模式下冻结 BN 层，否则默认为冻结状态
        self.freeze_bn = freeze_bn if self.training else True
        # 创建 BatchNorm1d 层
        self.bn = nn.BatchNorm1d(out_features, eps, momentum, True, True)
        # 创建权重的假量化模块
        self.weight_fake_quant = self.qconfig.weight()
        # 如果有偏置，则创建参数化偏置
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        # 重置 BN 层参数
        self.reset_bn_parameters()

        # 在调用 reset_bn_parameters 后调用此方法，
        # 因为它们会修改相同的状态
        if self.training:
            # 如果需要冻结 BN 层，则冻结统计信息
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

    def reset_running_stats(self):
        # 重置 BN 层运行统计信息
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        # 重置 BN 层参数
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)  # 初始化 BN 层权重为均匀分布
        init.zeros_(self.bn.bias)  # 初始化 BN 层偏置为零

    def reset_parameters(self):
        # 调用父类方法重置模块参数
        super().reset_parameters()

    def update_bn_stats(self):
        # 更新 BN 层统计信息
        self.freeze_bn = False  # 设置 BN 层非冻结状态
        self.bn.training = True  # 设置 BN 层训练模式为 True
        return self

    def freeze_bn_stats(self):
        # 冻结 BN 层统计信息
        self.freeze_bn = True  # 设置 BN 层冻结状态
        self.bn.training = False  # 设置 BN 层训练模式为 False
        return self
    def forward(self, input):
        assert self.bn.running_var is not None  # 断言确保批归一化的 running_var 已经计算好

        # 根据论文 https://arxiv.org/pdf/1806.08342.pdf 第18页的动机，
        # 使用批归一化的 running statistics 缩放线性层的权重，以减少权重抖动。
        #
        # 原先的实现是：
        #
        #   x1 = F.linear(x0, fq(w), b)
        #   x2 = self.bn(x1)
        #
        # 现在的实现是：
        #
        #   # 根据前一批次的 running statistics 缩放权重
        #   scale_factor = bn.w / bn.running_std_from_prev_batch
        #   # 在没有偏置的情况下进行线性变换
        #   x1_scaled = F.linear(x0, fq(w * scale_factor), 0)
        #   # 反向缩放并添加原始偏置
        #   x1_orig = x1_scaled / scale_factor + b
        #   x2 = self.bn(x1_orig)

        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)  # 计算批归一化的标准差
        scale_factor = self.bn.weight / running_std  # 计算缩放因子，用于缩放权重
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))  # 缩放权重
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)  # 创建与 self.bias 相同形状的零张量
        else:
            zero_bias = torch.zeros(self.out_features, device=scaled_weight.device)  # 创建全零张量，位于 scaled_weight 的设备上
        linear_out = F.linear(input, scaled_weight, zero_bias)  # 线性变换，使用缩放后的权重和零偏置
        linear_out_orig = linear_out / scale_factor.reshape(bias_shape)  # 反向缩放并恢复原始偏置
        if self.bias is not None:
            linear_out_orig = linear_out_orig + self.bias.reshape(bias_shape)  # 添加原始偏置
        bn_out = self.bn(linear_out_orig)  # 使用批归一化层处理线性变换结果
        return bn_out

    def train(self, mode=True):
        """
        批归一化层的训练行为受 self.training 标志控制。如果 BN 被冻结，
        确保调用 `model.train()` 后仍然保持其行为正确。
        """
        self.training = mode  # 设置当前模式（训练或评估）
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)  # 递归调用所有子模块的 train 方法，保持一致性
        return self

    @classmethod
    # 从一个浮点数模块或者 qparams_dict 创建一个量化训练模块

        # 参数:
        # - `mod`：一个浮点数模块，可以是由 torch.ao.quantization 工具或者用户直接创建的
        # - `use_precomputed_fake_quant`：是否使用预计算的伪量化，默认为 False

        # 断言输入的模块类型必须为 nni.LinearBn1d，否则抛出异常
        assert type(mod) == nni.LinearBn1d, 'qat.' + cls.__name__ + \
            '.from_float 只适用于 ' + nni.LinearBn1d.__name__
        # 断言输入的浮点数模块必须定义了 qconfig 属性
        assert hasattr(mod, 'qconfig'), '输入的浮点数模块必须定义 qconfig'
        # 断言 qconfig 属性不为空
        assert mod.qconfig, '输入的浮点数模块必须有有效的配置'

        # 获取模块的 qconfig
        qconfig = mod.qconfig
        # 分别获取模块中的 linear 和 bn
        linear, bn = mod[0], mod[1]

        # 根据 linear 和 bn 的参数创建一个新的量化训练模块
        qat_linearbn = cls(linear.in_features, linear.out_features, linear.bias is not None,
                           bn.eps, bn.momentum,
                           False, qconfig)

        # 将权重和偏置从浮点数模块复制到量化训练模块中对应的位置
        qat_linearbn.weight = linear.weight
        qat_linearbn.bias = linear.bias
        qat_linearbn.bn.weight = bn.weight
        qat_linearbn.bn.bias = bn.bias
        qat_linearbn.bn.running_mean = bn.running_mean
        qat_linearbn.bn.running_var = bn.running_var
        qat_linearbn.bn.num_batches_tracked = bn.num_batches_tracked

        # 返回创建的量化训练模块
        return qat_linearbn

    # 将当前的量化训练模块转换为等效的浮点数模块
    def to_float(self):

        # 创建一个新的 torch.nn.Linear 模块，使用当前模块的输入和输出特征数量
        linear = torch.nn.Linear(self.in_features, self.out_features)

        # 断言当前模块的 bn 属性中运行均值和方差不为空
        assert self.bn.running_var is not None and self.bn.running_mean is not None

        # 调用 fuse_linear_bn_weights 函数，将当前模块的权重、偏置、运行均值、运行方差等参数融合到新的 linear 模块中
        linear.weight, linear.bias = fuse_linear_bn_weights(
            self.weight,
            self.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps,
            self.bn.weight,
            self.bn.bias)

        # 返回转换后的浮点数模块
        return linear
```