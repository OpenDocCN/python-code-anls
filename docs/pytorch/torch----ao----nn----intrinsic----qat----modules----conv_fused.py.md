# `.\pytorch\torch\ao\nn\intrinsic\qat\modules\conv_fused.py`

```
# mypy: allow-untyped-defs
# 引入数学库和 PyTorch 库及其模块
import math
import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.nn.functional as F
# 从 torch.nn 中导入初始化方法和融合卷积批归一化权重的函数
from torch.nn import init
from torch.nn.utils import fuse_conv_bn_weights
# 从 torch.nn.modules.utils 中导入处理单、双、三维的工具函数
from torch.nn.modules.utils import _single, _pair, _triple
# 从 torch.nn.parameter 中导入参数类
from torch.nn.parameter import Parameter
# 从 typing 模块中导入类型变量 TypeVar
from typing import TypeVar

# 定义可导出的类名列表
__all__ = ['ConvBn1d', 'ConvBnReLU1d', 'ConvReLU1d', 'ConvBn2d', 'ConvBnReLU2d', 'ConvReLU2d', 'ConvBn3d',
           'ConvBnReLU3d', 'ConvReLU3d', 'update_bn_stats', 'freeze_bn_stats']
# 定义维度到批归一化类的映射关系字典
_BN_CLASS_MAP = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}

# 定义一个类型变量 MOD，其限制为 nn.modules.conv._ConvNd 的子类
MOD = TypeVar('MOD', bound=nn.modules.conv._ConvNd)

# 定义一个类 _ConvBnNd，继承自 nn.modules.conv._ConvNd 和 nni._FusedModule
class _ConvBnNd(nn.modules.conv._ConvNd, nni._FusedModule):

    _version = 2
    _FLOAT_MODULE = MOD

    # 初始化方法
    def __init__(self,
                 # ConvNd 参数
                 in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups,
                 bias,
                 padding_mode,
                 # BatchNormNd 参数
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # 本模块的参数
                 freeze_bn=False,
                 qconfig=None,
                 dim=2):
        # 调用父类 _ConvNd 的初始化方法
        nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, transposed,
                                         output_padding, groups, False, padding_mode)
        # 断言语句，确保 qconfig 参数不为空，用于量化训练
        assert qconfig, 'qconfig must be provided for QAT module'
        # 设置量化配置
        self.qconfig = qconfig
        # 如果模型处于训练状态，则根据 freeze_bn 参数决定是否冻结批归一化层的统计信息
        self.freeze_bn = freeze_bn if self.training else True
        # 创建批归一化层对象，根据维度 dim 选择不同的批归一化类
        self.bn = _BN_CLASS_MAP[dim](out_channels, eps, momentum, True, True)
        # 创建量化权重对象
        self.weight_fake_quant = self.qconfig.weight()
        # 如果有偏置，则创建偏置参数
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        # 重置批归一化层的参数
        self.reset_bn_parameters()

        # 在修改相同状态后需要调用此方法
        # 如果处于训练状态，则根据 freeze_bn 参数决定是否冻结批归一化层的统计信息，否则冻结统计信息
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

        # 是否启用更慢的路径来提高数值稳定性
        self._enable_slow_path_for_better_numerical_stability = False

    # 重置运行时统计信息
    def reset_running_stats(self):
        self.bn.reset_running_stats()

    # 重置批归一化层的参数
    def reset_bn_parameters(self):
        # 重置批归一化层的运行统计信息
        self.bn.reset_running_stats()
        # 初始化批归一化层的权重为均匀分布
        init.uniform_(self.bn.weight)
        # 初始化批归一化层的偏置为零
        init.zeros_(self.bn.bias)
        # 注意：以下实际上是针对卷积层的初始化，而非批归一化层
        if self.bias is not None:
            # 计算输入和输出通道数，并根据其计算初始化边界
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            # 初始化偏置参数为指定范围内的均匀分布
            init.uniform_(self.bias, -bound, bound)
    # 调用父类方法重置参数
    def reset_parameters(self):
        super().reset_parameters()

    # 更新 BatchNorm 的统计信息，允许训练
    def update_bn_stats(self):
        self.freeze_bn = False  # 解冻 BatchNorm
        self.bn.training = True  # 设置 BatchNorm 为训练模式
        return self

    # 冻结 BatchNorm 的统计信息，不允许训练
    def freeze_bn_stats(self):
        self.freeze_bn = True  # 冻结 BatchNorm
        self.bn.training = False  # 设置 BatchNorm 为非训练模式
        return self

    # 私有方法：根据启用的慢速路径或近似路径执行前向传播
    def _forward(self, input):
        if self._enable_slow_path_for_better_numerical_stability:
            return self._forward_slow(input)  # 使用慢速路径执行前向传播
        return self._forward_approximate(input)  # 使用近似路径执行前向传播

    # 近似方法：融合卷积和 BatchNorm 的近似方法，只需一次前向传播
    def _forward_approximate(self, input):
        """Approximated method to fuse conv and bn. It requires only one forward pass.
        conv_orig = conv / scale_factor where scale_factor = bn.weight / running_std
        """
        assert self.bn.running_var is not None
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)  # 计算运行时标准差
        scale_factor = self.bn.weight / running_std  # 计算缩放因子
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))  # 缩放权重
        # 在这里使用零偏置，因为原始卷积的偏置稍后会被添加
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias, dtype=input.dtype)
        else:
            zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device, dtype=input.dtype)
        conv = self._conv_forward(input, scaled_weight, zero_bias)  # 执行卷积操作
        conv_orig = conv / scale_factor.reshape(bias_shape)  # 对卷积结果进行反缩放
        if self.bias is not None:
            conv_orig = conv_orig + self.bias.reshape(bias_shape)  # 添加偏置
        conv = self.bn(conv_orig)  # 执行 BatchNorm 操作
        return conv

    # 返回额外的表示信息，当前为调用父类的 extra_repr 方法
    def extra_repr(self):
        # TODO(jerryzh): extend
        return super().extra_repr()

    # 前向传播方法，调用 _forward 方法执行
    def forward(self, input):
        return self._forward(input)

    # 重写 train 方法，控制模型的训练模式和 BatchNorm 的行为
    def train(self, mode=True):
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
        self.training = mode  # 设置模型的训练模式
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)  # 设置所有子模块的训练模式
        return self

    # ===== 序列化版本历史说明 =====
    #
    # Version 1/None
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- gamma : Tensor
    #   |--- beta : Tensor
    #   |--- running_mean : Tensor
    #   |--- running_var : Tensor
    #   |--- num_batches_tracked : Tensor
    #
    # Version 2
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- bn : Module
    #        |--- weight : Tensor (moved from v1.self.gamma)
    #        |--- bias : Tensor (moved from v1.self.beta)
    #        |--- running_mean : Tensor (moved from v1.self.running_mean)
    # 定义一个方法，用于加载模型状态字典中的参数和缓冲区
    # state_dict: 模型的状态字典，包含了模型的参数和缓冲区
    # prefix: 当前模块的前缀
    # local_metadata: 模型本地的元数据
    # strict: 是否启用严格模式，如果为True，则在state_dict中找不到对应键时会报错
    # missing_keys: 用于记录找不到的键的列表
    # unexpected_keys: 用于记录未预期的键的列表
    # error_msgs: 用于记录错误消息的列表
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # 获取模型版本号，如果未定义则默认为None
        version = local_metadata.get('version', None)
        # 如果版本号不存在或者为1，则执行以下代码
        if version is None or version == 1:
            # 定义v2到v1参数名的映射关系，将v2格式的参数名映射为v1格式的参数名
            v2_to_v1_names = {
                'bn.weight': 'gamma',                 # 权重参数映射
                'bn.bias': 'beta',                    # 偏置参数映射
                'bn.running_mean': 'running_mean',    # 移动平均值参数映射
                'bn.running_var': 'running_var',      # 移动方差参数映射
                'bn.num_batches_tracked': 'num_batches_tracked',  # 被追踪批次数映射
            }
            # 遍历映射关系字典
            for v2_name, v1_name in v2_to_v1_names.items():
                # 如果带有v1格式名称的参数在state_dict中存在
                if prefix + v1_name in state_dict:
                    # 将v2格式名称的参数赋值为v1格式名称的参数值，并从state_dict中移除v1格式名称的参数
                    state_dict[prefix + v2_name] = state_dict[prefix + v1_name]
                    state_dict.pop(prefix + v1_name)
                # 如果带有v2格式名称的参数在state_dict中存在
                elif prefix + v2_name in state_dict:
                    # 在某个时期，此模块的前向兼容性出现了问题，因此允许使用v2样式条目以修复这个前向兼容性问题
                    pass
                # 如果启用了严格模式且state_dict中找不到v2格式名称的参数
                elif strict:
                    # 将缺失的参数名加入到missing_keys列表中
                    missing_keys.append(prefix + v2_name)

        # 调用父类的_load_from_state_dict方法，继续加载模型状态字典中的其余部分
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
    # 创建一个量化训练模块（Quantization Aware Training Module），从给定的浮点数模块或qparams_dict中生成
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        # 断言输入的模块类型必须与预期的浮点数模块类型相同，否则抛出异常
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__  # type: ignore[attr-defined]
        # 断言输入的浮点数模块必须定义了qconfig属性
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        # 断言qconfig属性不能为空，确保输入的浮点数模块有一个有效的qconfig配置
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        
        # 获取输入浮点数模块的qconfig配置
        qconfig = mod.qconfig
        # 分别获取浮点数模块的第一个(conv)和第二个(bn)子模块
        conv, bn = mod[0], mod[1]
        
        # 使用cls类的构造函数创建一个量化训练模块，包含conv和bn的相关参数
        qat_convbn = cls(conv.in_channels, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, conv.dilation,
                         conv.groups, conv.bias is not None,
                         conv.padding_mode,
                         bn.eps, bn.momentum,
                         False,
                         qconfig)
        
        # 将量化训练模块的权重和偏置设置为与输入浮点数模块的对应参数相同
        qat_convbn.weight = conv.weight
        qat_convbn.bias = conv.bias
        qat_convbn.bn.weight = bn.weight
        qat_convbn.bn.bias = bn.bias
        qat_convbn.bn.running_mean = bn.running_mean
        qat_convbn.bn.running_var = bn.running_var
        # 设置量化训练模块的bn模块的num_batches_tracked属性与输入浮点数模块的相同
        # 由于类型推断问题，使用type: ignore[has-type]忽略类型检查错误
        qat_convbn.bn.num_batches_tracked = bn.num_batches_tracked  # type: ignore[has-type]
        
        # 返回创建的量化训练模块
        return qat_convbn
    def to_float(self):
        # 获取当前对象的类
        cls = type(self)
        # 使用类属性定义的浮点数转换模块，创建一个转换器对象
        conv = cls._FLOAT_CONV_MODULE(  # type: ignore[attr-defined]
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            self.padding_mode)
        # 将当前对象的权重参数作为转换器对象的权重参数
        conv.weight = torch.nn.Parameter(self.weight.detach())
        # 如果存在偏置，则将当前对象的偏置参数作为转换器对象的偏置参数
        if self.bias is not None:
            conv.bias = torch.nn.Parameter(self.bias.detach())

        # 如果定义了浮点数化的 BatchNorm 模块
        if cls._FLOAT_BN_MODULE:  # type: ignore[attr-defined]
            # 将 BatchNorm 的 running_mean 和 running_var 以及其它参数融合到卷积层中
            assert self.bn.running_var is not None and self.bn.running_mean is not None
            conv.weight, conv.bias = fuse_conv_bn_weights(
                conv.weight,
                conv.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
                self.bn.weight,
                self.bn.bias
            )

        # 如果定义了浮点数化的 ReLU 模块
        if cls._FLOAT_RELU_MODULE:  # type: ignore[attr-defined]
            # 创建一个包含卷积层和 ReLU 层的模块列表
            modules = []
            modules.append(conv)
            relu = cls._FLOAT_RELU_MODULE()  # type: ignore[attr-defined]
            modules.append(relu)
            # 将卷积层和 ReLU 层融合成一个单一的浮点数化模块
            conv_relu = cls._FUSED_FLOAT_MODULE(*modules)  # type: ignore[attr-defined]
            # 设置模块处于训练状态
            conv_relu.train(self.training)
            return conv_relu
        else:
            # 如果没有定义浮点数化的 ReLU 模块，则直接将卷积层设置为训练状态
            conv.train(self.training)
            return conv
class ConvBn1d(_ConvBnNd, nn.Conv1d):
    r"""
    A ConvBn1d module is a module fused from Conv1d and BatchNorm1d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv1d` and
    :class:`torch.nn.BatchNorm1d`.

    Similar to :class:`torch.nn.Conv1d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
            A boolean indicating whether to freeze BatchNorm statistics during training.
        weight_fake_quant:
            FakeQuantize module used for quantizing the weights.

    """
    _FLOAT_BN_MODULE = nn.BatchNorm1d
    _FLOAT_RELU_MODULE: None = None
    _FLOAT_MODULE = nni.ConvBn1d
    _FLOAT_CONV_MODULE = nn.Conv1d

    def __init__(self,
                 # Conv1d args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None,
                 padding_mode='zeros',
                 # BatchNorm1d args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        _ConvBnNd.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, False, _single(0), groups, bias, padding_mode,
                           eps, momentum, freeze_bn, qconfig, dim=1)

class ConvBnReLU1d(ConvBn1d):
    r"""
    A ConvBnReLU1d module is a module fused from Conv1d, BatchNorm1d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv1d` and
    :class:`torch.nn.BatchNorm1d` and :class:`torch.nn.ReLU`.

    Similar to `torch.nn.Conv1d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant:
            FakeQuantize module used for quantizing the weights.

    """
    # base class defines _FLOAT_MODULE as "ConvBn1d"
    _FLOAT_MODULE = nni.ConvBnReLU1d  # type: ignore[assignment]
    _FLOAT_CONV_MODULE = nn.Conv1d
    _FLOAT_BN_MODULE = nn.BatchNorm1d
    _FLOAT_RELU_MODULE = nn.ReLU  # type: ignore[assignment]
    # module class after fusing bn into conv
    _FUSED_FLOAT_MODULE = nni.ConvReLU1d


注释：
    # 初始化函数，设置了Conv1d和BatchNorm1d的参数以及当前模块特有的参数
    def __init__(self,
                 # Conv1d args：卷积层的参数
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None,
                 padding_mode='zeros',
                 # BatchNorm1d args：批标准化层的参数
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module：当前模块特有的参数
                 freeze_bn=False,  # 是否冻结批标准化层
                 qconfig=None):     # 量化配置

        # 调用父类的初始化函数，传递所有的参数
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias,
                         padding_mode, eps, momentum,
                         freeze_bn,  # 冻结批标准化层的标志
                         qconfig)     # 量化配置

    # 前向传播函数，使用ReLU激活函数对卷积批标准化层的结果进行非线性处理
    def forward(self, input):
        return F.relu(ConvBn1d._forward(self, input))

    # 类方法，用于从浮点数模型中创建当前量化模型
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant)
# 定义一个类 ConvReLU1d，继承自 nn.Conv1d 和 nni._FusedModule，用于量化感知训练，将 Conv1d 和 ReLU 功能融合到一起。
class ConvReLU1d(nnqat.Conv1d, nni._FusedModule):
    r"""A ConvReLU1d module is a fused module of Conv1d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv1d` and
    :class:`~torch.nn.BatchNorm1d`.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    # 定义一个类变量，指向 nn.Conv1d 类型的模块
    _FLOAT_MODULE = nni.ConvReLU1d  # type: ignore[assignment]
    # 定义一个类变量，指向 nn.Conv1d 类型的模块
    _FLOAT_CONV_MODULE = nn.Conv1d
    # 定义一个类变量，表示没有 BatchNorm1d 模块
    _FLOAT_BN_MODULE: None = None
    # 定义一个类变量，指向 nn.ReLU 类型的模块
    _FLOAT_RELU_MODULE = nn.ReLU

    # 初始化方法
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 qconfig=None):
        # 调用父类的初始化方法，初始化 Conv1d 相关参数
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode,
                         qconfig=qconfig)
        # 断言确保 qconfig 参数被提供，用于量化感知训练
        assert qconfig, 'qconfig must be provided for QAT module'
        # 将 qconfig 存储到当前对象的属性中
        self.qconfig = qconfig
        # 通过 qconfig 创建权重的 fake quant 模块，并存储到对象的属性中
        self.weight_fake_quant = self.qconfig.weight()

    # 前向传播方法
    def forward(self, input):
        # 执行卷积操作，并将结果传递给 ReLU 激活函数
        return F.relu(
            self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias))

    # 从浮点数模型转换为量化感知模型的类方法
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)

# 定义一个类 ConvBn2d，继承自 _ConvBnNd 和 nn.Conv2d，用于量化感知训练，将 Conv2d 和 BatchNorm2d 功能融合到一起。
class ConvBn2d(_ConvBnNd, nn.Conv2d):
    r"""
    A ConvBn2d module is a module fused from Conv2d and BatchNorm2d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.

    Similar to :class:`torch.nn.Conv2d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """
    # 定义一个类变量，指向 nni.ConvBn2d 类型的模块
    _FLOAT_MODULE = nni.ConvBn2d
    # 定义一个类变量，指向 nn.Conv2d 类型的模块
    _FLOAT_CONV_MODULE = nn.Conv2d
    # 定义一个类变量，指向 nn.BatchNorm2d 类型的模块
    _FLOAT_BN_MODULE = nn.BatchNorm2d
    # 定义一个类变量，表示没有 ReLU 模块
    _FLOAT_RELU_MODULE: None = None
    def __init__(self,
                 # ConvNd args
                 # 初始化函数，用于构造一个卷积层与批归一化层组合的模块
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None,
                 padding_mode='zeros',
                 # BatchNorm2d args
                 # 批归一化层的参数，其中 num_features 对应于 out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 # 冻结批归一化层的标志，默认为 False
                 freeze_bn=False,
                 # 量化配置，默认为 None
                 qconfig=None):
        
        # 将 kernel_size 转换为二元组
        kernel_size = _pair(kernel_size)
        # 将 stride 转换为二元组
        stride = _pair(stride)
        # 将 padding 转换为二元组
        padding = _pair(padding)
        # 将 dilation 转换为二元组
        dilation = _pair(dilation)
        
        # 调用父类 _ConvBnNd 的初始化方法，构建卷积与批归一化层组合的模块
        _ConvBnNd.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, False, _pair(0), groups, bias, padding_mode,
                           eps, momentum, freeze_bn, qconfig, dim=2)
# 定义 ConvReLU2d 类，继承自 nnqat.Conv2d 和 nni._FusedModule
class ConvReLU2d(nnqat.Conv2d, nni._FusedModule):
    r"""A ConvReLU2d module is a fused module of Conv2d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv2d` and
    :class:`~torch.nn.BatchNorm2d`.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    # _FLOAT_MODULE 类属性，用于指定浮点模型中的 ConvReLU2d 类
    _FLOAT_MODULE = nni.ConvReLU2d  # type: ignore[assignment]
    # _FLOAT_CONV_MODULE 类属性，用于指定浮点模型中的卷积层
    _FLOAT_CONV_MODULE = nn.Conv2d
    # _FLOAT_BN_MODULE 类属性，此处为 None，表示在融合模块中不使用 BatchNorm2d
    _FLOAT_BN_MODULE: None = None
    # _FLOAT_RELU_MODULE 类属性，用于指定浮点模型中的 ReLU 激活层
    _FLOAT_RELU_MODULE = nn.ReLU
    # 初始化函数，用于设置量化卷积层的参数和配置
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 qconfig=None):
        # 调用父类的初始化方法，设置卷积层的基本参数
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode,
                         qconfig=qconfig)
        # 断言确保 qconfig 参数不为空，因为量化训练需要提供 qconfig
        assert qconfig, 'qconfig must be provided for QAT module'
        # 将 qconfig 参数设置为当前对象的 qconfig 属性
        self.qconfig = qconfig
        # 使用 qconfig 中的 weight 方法来初始化 weight_fake_quant 属性
        self.weight_fake_quant = self.qconfig.weight()

    # 前向传播函数，实现量化卷积层的前向计算
    def forward(self, input):
        # 调用 F.relu 函数，对卷积操作的输出进行 ReLU 激活
        return F.relu(
            # 调用 _conv_forward 方法执行卷积操作，同时对权重进行量化
            self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias))

    # 类方法，用于从浮点模型转换为量化模型
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 调用父类的 from_float 方法，从浮点模型中创建量化模型
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)
class ConvBn3d(_ConvBnNd, nn.Conv3d):
    r"""
    A ConvBn3d module is a module fused from Conv3d and BatchNorm3d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv3d` and
    :class:`torch.nn.BatchNorm3d`.

    Similar to :class:`torch.nn.Conv3d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = nni.ConvBn3d
    _FLOAT_CONV_MODULE = nn.Conv3d
    _FLOAT_BN_MODULE = nn.BatchNorm3d
    _FLOAT_RELU_MODULE: None = None

    def __init__(
        self,
        # ConvNd args
        in_channels,          # 输入通道数
        out_channels,         # 输出通道数
        kernel_size,          # 卷积核大小
        stride=1,             # 步长，默认为1
        padding=0,            # 填充大小，默认为0
        dilation=1,           # 膨胀率，默认为1
        groups=1,             # 分组卷积，默认为1
        bias=None,            # 是否使用偏置，默认为None
        padding_mode="zeros", # 填充模式，默认为"zeros"
        # BatchNorm3d args
        # num_features: out_channels
        eps=1e-05,            # BatchNorm3d的epsilon值，默认为1e-05
        momentum=0.1,         # BatchNorm3d的动量，默认为0.1
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,      # 是否冻结BatchNorm，默认为False
        qconfig=None,         # 量化配置，默认为None
    ):
        kernel_size = _triple(kernel_size)   # 将kernel_size转换为三元组形式
        stride = _triple(stride)             # 将stride转换为三元组形式
        padding = _triple(padding)           # 将padding转换为三元组形式
        dilation = _triple(dilation)         # 将dilation转换为三元组形式
        _ConvBnNd.__init__(                  # 调用父类_ConvBnNd的初始化方法
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,                          # 是否使用ReLU，默认为False
            _triple(0),                     # ReLU的填充，默认为三元组(0, 0, 0)
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            qconfig,
            dim=3,                          # 维度为3，表示3维卷积
        )

class ConvBnReLU3d(ConvBn3d):
    r"""
    A ConvBnReLU3d module is a module fused from Conv3d, BatchNorm3d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv3d` and
    :class:`torch.nn.BatchNorm3d` and :class:`torch.nn.ReLU`.

    Similar to `torch.nn.Conv3d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = nni.ConvBnReLU3d  # type: ignore[assignment]
    _FLOAT_CONV_MODULE = nn.Conv3d
    _FLOAT_BN_MODULE = nn.BatchNorm3d
    _FLOAT_RELU_MODULE = nn.ReLU  # type: ignore[assignment]
    # module class after fusing bn into conv
    _FUSED_FLOAT_MODULE = nni.ConvReLU3d

    def __init__(
        self,
        # Conv3d args
        in_channels,          # 输入通道数
        out_channels,         # 输出通道数
        kernel_size,          # 卷积核大小
        stride=1,             # 步长，默认为1
        padding=0,            # 填充大小，默认为0
        dilation=1,           # 膨胀率，默认为1
        groups=1,             # 分组卷积，默认为1
        bias=None,            # 是否使用偏置，默认为None
        padding_mode="zeros", # 填充模式，默认为"zeros"
        # BatchNorm3d args
        # num_features: out_channels
        eps=1e-05,            # BatchNorm3d的epsilon值，默认为1e-05
        momentum=0.1,         # BatchNorm3d的动量，默认为0.1
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,      # 是否冻结BatchNorm，默认为False
        qconfig=None,         # 量化配置，默认为None
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            qconfig,
        )


# 调用父类的构造函数，初始化 ConvBn3d 对象
super().__init__(
    in_channels,        # 输入通道数
    out_channels,       # 输出通道数
    kernel_size,        # 卷积核大小
    stride,             # 步幅大小
    padding,            # 填充大小
    dilation,           # 空洞卷积的扩张大小
    groups,             # 分组卷积中的组数
    bias,               # 是否使用偏置项
    padding_mode,       # 填充模式
    eps,                # BatchNorm 的 epsilon 参数
    momentum,           # BatchNorm 的动量参数
    freeze_bn,          # 是否冻结 BatchNorm 参数
    qconfig,            # 量化配置
)

def forward(self, input):
    # 调用 ConvBn3d 的 _forward 方法，并对输出应用 ReLU 激活函数
    return F.relu(ConvBn3d._forward(self, input))

@classmethod
def from_float(cls, mod, use_precomputed_fake_quant=False):
    # 调用父类的 from_float 方法，返回从浮点模型转换而来的量化模型
    return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)
class ConvReLU3d(nnqat.Conv3d, nni._FusedModule):
    r"""A ConvReLU3d module is a fused module of Conv3d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv3d` and
    :class:`~torch.nn.BatchNorm3d`.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = nni.ConvReLU3d  # type: ignore[assignment]
    _FLOAT_CONV_MODULE = nn.Conv3d
    _FLOAT_BN_MODULE: None = None
    _FLOAT_RELU_MODULE = nn.ReLU

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        # 调用父类的初始化方法，设置卷积相关参数
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )
        # 断言确保在量化感知训练模块中必须提供 qconfig 参数
        assert qconfig, "qconfig must be provided for QAT module"
        # 设置量化配置
        self.qconfig = qconfig
        # 使用 qconfig 中的权重量化模块作为 weight_fake_quant 属性
        self.weight_fake_quant = self.qconfig.weight()

    def forward(self, input):
        # 执行前向传播，包括卷积操作和 ReLU 激活函数
        return F.relu(
            self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 调用父类的方法将 float 模型转换为量化感知训练模块
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)

def update_bn_stats(mod):
    # 如果 mod 是以下类型之一，则调用其更新批归一化统计信息的方法
    if type(mod) in {ConvBnReLU1d, ConvBnReLU2d, ConvBnReLU3d, ConvBn1d, ConvBn2d, ConvBn3d}:
        mod.update_bn_stats()

def freeze_bn_stats(mod):
    # 如果 mod 是以下类型之一，则调用其冻结批归一化统计信息的方法
    if type(mod) in {ConvBnReLU1d, ConvBnReLU2d, ConvBnReLU3d, ConvBn1d, ConvBn2d, ConvBn3d}:
        mod.freeze_bn_stats()
```