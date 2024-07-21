# `.\pytorch\torch\ao\nn\quantized\modules\normalization.py`

```py
# 引入torch模块，用于量化操作
import torch

# 定义公开的类列表
__all__ = ['LayerNorm', 'GroupNorm', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d']

# 继承自torch.nn.LayerNorm的量化版本LayerNorm类
class LayerNorm(torch.nn.LayerNorm):
    r"""This is the quantized version of :class:`~torch.nn.LayerNorm`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    
    # 初始化函数
    def __init__(self, normalized_shape, weight, bias, scale, zero_point, eps=1e-5,
                 elementwise_affine=True, device=None, dtype=None) -> None:
        # 工厂关键字参数
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类的初始化方法
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine,
                         **factory_kwargs)
        # 设置权重和偏置
        self.weight = weight
        self.bias = bias
        # 注册缓冲区，存储量化参数scale和zero_point
        self.register_buffer('scale', torch.tensor(scale, **factory_kwargs))
        self.register_buffer('zero_point', torch.tensor(zero_point, **factory_kwargs))

    # 前向传播方法
    def forward(self, input):
        # 调用torch.ops.quantized.layer_norm执行量化LayerNorm操作
        return torch.ops.quantized.layer_norm(
            input, self.normalized_shape, weight=self.weight, bias=self.bias,
            eps=self.eps, output_scale=self.scale, output_zero_point=self.zero_point)

    # 获取类名方法
    def _get_name(self):
        return 'QuantizedLayerNorm'

    # 从浮点数模型转换方法
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 计算量化参数scale和zero_point
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        # 创建新的QuantizedLayerNorm对象
        new_mod = cls(
            mod.normalized_shape, mod.weight, mod.bias, float(scale),
            int(zero_point), mod.eps, mod.elementwise_affine)
        return new_mod

    # 从参考模型转换方法
    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        # 创建新的QuantizedLayerNorm对象
        return cls(
            mod.normalized_shape, mod.weight, mod.bias, float(scale),
            int(zero_point), mod.eps, mod.elementwise_affine)

# 继承自torch.nn.GroupNorm的量化版本GroupNorm类
class GroupNorm(torch.nn.GroupNorm):
    r"""This is the quantized version of :class:`~torch.nn.GroupNorm`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    # 常量列表
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']

    # 初始化函数
    def __init__(self, num_groups, num_channels, weight, bias, scale, zero_point, eps=1e-5,
                 affine=True, device=None, dtype=None) -> None:
        # 工厂关键字参数
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类的初始化方法
        super().__init__(num_groups, num_channels, eps, affine, **factory_kwargs)
        # 设置权重和偏置
        self.weight = weight
        self.bias = bias
        # 注册缓冲区，存储量化参数scale和zero_point
        self.register_buffer('scale', torch.tensor(scale, **factory_kwargs))
        self.register_buffer('zero_point', torch.tensor(zero_point, **factory_kwargs))

    # 前向传播方法
    def forward(self, input):
        # 调用torch.ops.quantized.group_norm执行量化GroupNorm操作
        return torch.ops.quantized.group_norm(
            input, self.num_groups, self.weight, self.bias, self.eps, self.scale,
            self.zero_point)

    # 获取类名方法
    def _get_name(self):
        return 'QuantizedGroupNorm'

    # 类方法：从浮点数模型转换
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 暂未提供完整代码段，可以继续添加根据实际需要
    # 定义一个类方法 `from_float`，用于从一个量化模块 `mod` 转换为当前类的实例
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 调用 `activation_post_process` 的方法 `calculate_qparams()` 计算量化参数 `scale` 和 `zero_point`
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        # 使用计算得到的 `scale` 和 `zero_point` 创建一个新的当前类实例 `new_mod`
        new_mod = cls(
            mod.num_groups, mod.num_channels, mod.weight, mod.bias, float(scale), int(zero_point),
            mod.eps, mod.affine)
        # 返回创建的新实例 `new_mod`
        return new_mod
class InstanceNorm1d(torch.nn.InstanceNorm1d):
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm1d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    def __init__(self, num_features, weight, bias, scale, zero_point,
                 eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类构造函数初始化实例规范化层
        super().__init__(num_features, eps, momentum, affine, track_running_stats, **factory_kwargs)
        self.weight = weight
        self.bias = bias
        # 注册缓冲区 'scale' 和 'zero_point' 用于量化输出
        self.register_buffer('scale', torch.tensor(scale, **factory_kwargs))
        self.register_buffer('zero_point', torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        # 调用量化的实例规范化操作
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale,
            self.zero_point)

    def _get_name(self):
        # 返回当前类的名称字符串
        return 'QuantizedInstanceNorm1d'

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 获取量化参数 'scale' 和 'zero_point'，用于从浮点模型创建新的量化模型
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_features, mod.weight, mod.bias, float(scale), int(zero_point),
            mod.eps, mod.affine)
        return new_mod

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        # 从参考模型和给定的量化参数创建新的量化模型
        return cls(
            mod.num_features, mod.weight, mod.bias, float(scale), int(zero_point),
            mod.eps, mod.affine)

class InstanceNorm2d(torch.nn.InstanceNorm2d):
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm2d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    def __init__(self, num_features, weight, bias, scale, zero_point,
                 eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类构造函数初始化实例规范化层
        super().__init__(num_features, eps, momentum, affine, track_running_stats, **factory_kwargs)
        self.weight = weight
        self.bias = bias
        # 注册缓冲区 'scale' 和 'zero_point' 用于量化输出
        self.register_buffer('scale', torch.tensor(scale, **factory_kwargs))
        self.register_buffer('zero_point', torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        # 调用量化的实例规范化操作
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale,
            self.zero_point)

    def _get_name(self):
        # 返回当前类的名称字符串
        return 'QuantizedInstanceNorm2d'

    @classmethod
    # 类方法：从浮点数模型转换而来
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 获取模型的量化参数：缩放因子和零点
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        # 使用提取的量化参数创建新的量化模型对象
        new_mod = cls(
            mod.num_features, mod.weight, mod.bias, float(scale), int(zero_point),
            mod.eps, mod.affine)
        # 返回新创建的量化模型对象
        return new_mod

    # 类方法：从参考模型转换而来
    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        # 使用指定的量化参数创建新的量化模型对象
        return cls(
            mod.num_features, mod.weight, mod.bias, float(scale), int(zero_point),
            mod.eps, mod.affine)
# 定义一个继承自 torch.nn.InstanceNorm3d 的类 InstanceNorm3d
# 这是 torch.nn.InstanceNorm3d 的量化版本

# 添加了额外的参数：
# * **scale** - 输出的量化尺度，类型为双精度浮点数
# * **zero_point** - 输出的量化零点，类型为长整型

class InstanceNorm3d(torch.nn.InstanceNorm3d):
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm3d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    
    # 初始化函数，接受多个参数
    def __init__(self, num_features, weight, bias, scale, zero_point,
                 eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False, device=None, dtype=None) -> None:
        
        # 创建一个关键字参数字典，包括 device 和 dtype
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # 调用父类的初始化方法，传入 num_features, eps, momentum, affine, track_running_stats 等参数
        super().__init__(num_features, eps, momentum, affine, track_running_stats, **factory_kwargs)
        
        # 设置权重和偏置
        self.weight = weight
        self.bias = bias
        
        # 将 scale 和 zero_point 注册为缓冲区（buffer），使用 torch.tensor 创建张量并存储
        self.register_buffer('scale', torch.tensor(scale, **factory_kwargs))
        self.register_buffer('zero_point', torch.tensor(zero_point, **factory_kwargs))

    # 前向传播函数，接受输入 input
    def forward(self, input):
        # 调用 torch.ops.quantized.instance_norm 执行量化实例归一化操作
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale,
            self.zero_point)

    # 返回模块的名称字符串
    def _get_name(self):
        return 'QuantizedInstanceNorm3d'

    # 类方法，从浮点模型 mod 转换为量化模型
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 计算量化参数 scale 和 zero_point
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        # 创建一个新的量化实例归一化模型 new_mod
        new_mod = cls(
            mod.num_features, mod.weight, mod.bias, float(scale), int(zero_point),
            mod.eps, mod.affine)
        return new_mod

    # 类方法，从参考模型 mod 和给定的 scale 和 zero_point 创建量化模型
    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(
            mod.num_features, mod.weight, mod.bias, float(scale), int(zero_point),
            mod.eps, mod.affine)
```