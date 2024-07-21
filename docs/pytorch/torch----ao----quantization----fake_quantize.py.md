# `.\pytorch\torch\ao\quantization\fake_quantize.py`

```py
# 引入 torch 库，实现了张量和模块的操作
# 引入 Module 类，用于定义神经网络模块的基类
import torch
from torch.nn import Module
# 从 torch.ao.quantization.observer 模块中引入多个观察者类
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    HistogramObserver,
    MovingAveragePerChannelMinMaxObserver,
    FixedQParamsObserver,
    default_fixed_qparams_range_0to1_observer,
    default_fixed_qparams_range_neg1to1_observer,
    _with_args,
)
# 引入正则表达式模块 re
import re
# 从 abc 模块中引入 ABC 抽象基类和 abstractmethod 装饰器
from abc import ABC, abstractmethod
# 从 typing 模块中引入 Any 和 Tuple 类型
from typing import Any, Tuple

# 模块中导出的公共接口列表
__all__ = [
    "FakeQuantizeBase",
    "FakeQuantize",
    "FixedQParamsFakeQuantize",
    "FusedMovingAvgObsFakeQuantize",
    "disable_fake_quant",
    "disable_observer",
    "enable_fake_quant",
    "enable_observer",
    "default_fake_quant",
    "default_weight_fake_quant",
    "default_dynamic_fake_quant",
    "default_fixed_qparams_range_neg1to1_fake_quant",
    "default_fixed_qparams_range_0to1_fake_quant",
    "default_symmetric_fixed_qparams_fake_quant",
    "default_affine_fixed_qparams_fake_quant",
    "default_per_channel_weight_fake_quant",
    "default_embedding_fake_quant",
    "default_embedding_fake_quant_4bit",
    "default_histogram_fake_quant",
    "default_fused_act_fake_quant",
    "default_fused_wt_fake_quant",
    "default_fused_per_channel_wt_fake_quant",
    "fused_wt_fake_quant_range_neg_127_to_127",
    "fused_per_channel_wt_fake_quant_range_neg_127_to_127",
]

# 判断量化方案是否为按通道量化
def _is_per_channel(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_channel_symmetric, torch.per_channel_affine, torch.per_channel_affine_float_qparams]

# 判断量化方案是否为按张量量化
def _is_per_tensor(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]

# 判断量化方案是否为对称量化
def _is_symmetric_quant(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_channel_symmetric]

# 判断量化方案是否为浮点数参数量化
def _is_float_qparams(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_channel_affine_float_qparams, ]

# 定义一个抽象基类 FakeQuantizeBase，继承自 ABC 和 Module
class FakeQuantizeBase(ABC, Module):
    r"""Base fake quantize module.

    Base fake quantize module
    Any fake quantize implementation should derive from this class.

    Concrete fake quantize module should follow the same API. In forward, they will update
    the statistics of the observed Tensor and fake quantize the input. They should also provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.

    """

    # 定义两个缓冲区属性 fake_quant_enabled 和 observer_enabled，类型为 uint8 的张量
    # 用于支持在分布式数据并行（DDP）中的复制，因为 NCCL 不支持布尔张量
    fake_quant_enabled: torch.Tensor
    observer_enabled: torch.Tensor

    # 构造方法，初始化 fake_quant_enabled 和 observer_enabled 属性
    def __init__(self):
        """Set fake_quant_enabled and observer_enabled."""
        # 调用父类的构造方法
        super().__init__()
        # 注册两个缓冲区属性 fake_quant_enabled 和 observer_enabled
        # 使用 torch.tensor 创建 uint8 类型的张量，初始值为 [1]
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('observer_enabled', torch.tensor([1], dtype=torch.uint8))
    # 声明一个抽象方法，需要在子类中实现具体逻辑
    @abstractmethod
    def forward(self, x):
        pass

    # 声明一个抽象方法，需要在子类中实现具体逻辑，用于计算量化参数
    @abstractmethod
    def calculate_qparams(self, **kwargs):
        pass

    # 导出的方法，用于启用或禁用伪量化功能
    @torch.jit.export
    def enable_fake_quant(self, enabled: bool = True) -> None:
        self.fake_quant_enabled[0] = 1 if enabled else 0

    # 导出的方法，用于禁用伪量化功能，实际上调用了 enable_fake_quant(False)
    @torch.jit.export
    def disable_fake_quant(self):
        self.enable_fake_quant(False)

    # 导出的方法，用于启用或禁用观察者功能
    @torch.jit.export
    def enable_observer(self, enabled: bool = True) -> None:
        self.observer_enabled[0] = 1 if enabled else 0

    # 导出的方法，用于禁用观察者功能，实际上调用了 enable_observer(False)
    @torch.jit.export
    def disable_observer(self):
        self.enable_observer(False)

    # 类方法，通过指定的关键字参数返回一个带有参数的 fake_quantize 构造器
    @classmethod
    def with_args(cls, **kwargs):
        fake_quant_constructor = _with_args(cls, **kwargs)
        # 需要将 fake_quantize 的模块设置为 "torch.ao.quantization.fake_quantize"，以满足公共和私有的需求
        fake_quant_constructor.__module__ = "torch.ao.quantization.fake_quantize"
        return fake_quant_constructor
class FakeQuantize(FakeQuantizeBase):
    r"""Simulate the quantize and dequantize operations in training time.

    The output of this module is given by::

        x_out = (
          clamp(round(x/scale + zero_point), quant_min, quant_max) - zero_point
        ) * scale

    * :attr:`is_dynamic` indicates whether the fake quantie is a placeholder for dynamic quantization
      operators (choose_qparams -> q -> dq) or static quantization operators (q -> dq)

    * :attr:`scale` defines the scale factor used for quantization.

    * :attr:`zero_point` specifies the quantized value to which 0 in floating point maps to

    * :attr:`fake_quant_enabled` controls the application of fake quantization on tensors, note that
      statistics can still be updated.

    * :attr:`observer_enabled` controls statistics collection on tensors

    * :attr:`dtype` specifies the quantized dtype that is being emulated with fake-quantization,
        allowable values are torch.qint8 and torch.quint8.

    Args:

        observer (module): Module for observing statistics on input tensors and calculating scale
          and zero-point.
        observer_kwargs (optional): Arguments for the observer module

    Attributes:
        activation_post_process (Module): User provided module that collects statistics on the input tensor and
          provides a method to calculate scale and zero-point.

    """

    # 定义了一个名为 FakeQuantize 的类，继承自 FakeQuantizeBase 类
    # 该类用于在训练时模拟量化和反量化操作

    scale: torch.Tensor
    zero_point: torch.Tensor
    # 类的属性包括 scale 和 zero_point，都是 torch.Tensor 类型
    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=None, quant_max=None, is_dynamic=False, **observer_kwargs):
        super().__init__()
        # 如果 quant_min 和 quant_max 都不为 None，则将它们添加到 observer_kwargs 中
        if quant_min is not None and quant_max is not None:
            # 断言 quant_min 必须小于等于 quant_max
            assert quant_min <= quant_max, \
                'quant_min must be less than or equal to quant_max'
            
            # 检查 observer 是否具有属性 "p"，如果是 _PartialWrapper 类型，则可能在 observer.p.keywords["dtype"] 中存储 dtype
            dtype = observer_kwargs.get("dtype", torch.quint8)
            if hasattr(observer, "p"):
                # 在这种情况下，dtype 可能存储在 observer.p.keywords["dtype"] 中
                dtype = getattr(getattr(observer, "p", {}), "keywords", {}).get(
                    "dtype", dtype
                )
            
            # 断言 quant_min 在 dtype 的范围内
            assert torch.iinfo(dtype).min <= quant_min, 'quant_min out of bound'
            # 断言 quant_max 在 dtype 的范围内
            assert quant_max <= torch.iinfo(dtype).max, 'quant_max out of bound'
            
            # 更新 observer_kwargs，添加 quant_min 和 quant_max
            observer_kwargs.update({"quant_min": quant_min, "quant_max": quant_max})
        
        # 将 is_dynamic 设置为 observer_kwargs 中的值
        observer_kwargs["is_dynamic"] = is_dynamic
        
        # 创建 activation_post_process 实例，使用给定的观察器和参数
        self.activation_post_process = observer(**observer_kwargs)
        
        # TODO: 保留 self.quant_min/max 以保持向后兼容；在几个发布后删除
        # 用户应该使用 self.activation_post_process.quant_min
        # 设置 self.quant_min 和 self.quant_max
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        self.is_dynamic = self.activation_post_process.is_dynamic
        
        # 根据 activation_post_process 的量化方案确定 zero_point_dtype
        if _is_float_qparams(self.activation_post_process.qscheme):
            zero_point_dtype = torch.float
        else:
            zero_point_dtype = torch.int
        
        # 注册缓冲区 'scale' 和 'zero_point'
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=zero_point_dtype))
        
        # 设置 self.dtype 和 self.qscheme
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        
        # 如果 activation_post_process 具有属性 'ch_axis'，则设置 self.ch_axis；否则设置为 -1
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1
        
        # 断言量化方案必须是 per channel 或 per tensor，fake quantize 只支持这两种方案
        assert _is_per_channel(self.qscheme) or \
            _is_per_tensor(self.qscheme), \
            'Only per channel and per tensor quantization are supported in fake quantize' + \
            ' got qscheme: ' + str(self.qscheme)
        
        # 设置 self.is_per_channel 标志
        self.is_per_channel = _is_per_channel(self.qscheme)

    @torch.jit.export
    def calculate_qparams(self):
        # 调用 activation_post_process 的 calculate_qparams 方法，并返回结果
        return self.activation_post_process.calculate_qparams()
    # 定义前向传播方法，接收输入 X
    def forward(self, X):
        # 如果观察器启用且第一个元素为1
        if self.observer_enabled[0] == 1:
            # 调用激活后处理方法处理输入 X 的数据
            self.activation_post_process(X.detach())
            # 计算量化参数 _scale 和 _zero_point
            _scale, _zero_point = self.calculate_qparams()
            # 将计算得到的 _scale 和 _zero_point 转移到对应设备
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            # 如果当前 scale 的形状与 _scale 不同，调整 scale 和 zero_point 的大小
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            # 复制 _scale 和 _zero_point 到 scale 和 zero_point
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        # 如果伪量化启用且第一个元素为1
        if self.fake_quant_enabled[0] == 1:
            # 如果是按通道量化
            if self.is_per_channel:
                # 对 X 进行按通道伪量化
                X = torch.fake_quantize_per_channel_affine(
                    X, self.scale, self.zero_point,
                    self.ch_axis, self.activation_post_process.quant_min, self.activation_post_process.quant_max)
            else:
                # 对 X 进行整体伪量化
                X = torch.fake_quantize_per_tensor_affine(
                    X, self.scale, self.zero_point,
                    self.activation_post_process.quant_min, self.activation_post_process.quant_max)
        
        # 返回量化后的输入 X
        return X

    @torch.jit.export
    def extra_repr(self):
        # 返回对象的额外表示，包括各种量化相关的属性
        return f'fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, ' \
               f'quant_min={self.activation_post_process.quant_min}, quant_max={self.activation_post_process.quant_max}, ' \
               f'dtype={self.dtype}, qscheme={self.qscheme}, ch_axis={self.ch_axis}, ' \
               f'scale={self.scale}, zero_point={self.zero_point}'

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # 由于目前不能将标量值注册为缓冲区，需要手动指定序列化过程
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # 将 scale 和 zero_point 存储到状态字典中
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point
    # 从给定的状态字典 `state_dict` 中加载模型的部分参数。
    # `prefix` 是当前模型参数在 `state_dict` 中的前缀。
    # `local_metadata` 是模型本地的元数据。
    # `strict` 控制是否严格匹配参数名称。
    # `missing_keys` 是一个列表，用于记录未找到的参数键。
    # `unexpected_keys` 是一个列表，用于记录未预期的参数键。
    # `error_msgs` 是一个用于记录错误信息的列表。
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 定义需要特殊处理的本地状态参数列表
        local_state = ['scale', 'zero_point']
        # 遍历本地状态参数列表
        for name in local_state:
            # 构建完整的参数键名
            key = prefix + name
            # 检查 `state_dict` 中是否存在该参数键
            if key in state_dict:
                # 如果存在，则获取对应的值
                val = state_dict[key]
                # 自定义处理，允许加载 `scale` 和 `zero_point`
                # 如果当前参数是 `scale`
                if name == 'scale':
                    # 调整当前对象的 `scale` 属性大小并赋值
                    self.scale.resize_(val.shape)
                else:
                    # 否则当前参数是 `zero_point`
                    assert name == 'zero_point'
                    # 调整当前对象的 `zero_point` 属性大小并赋值
                    self.zero_point.resize_(val.shape)
                # 对于 torchscript 模块，需要在这里更新属性值，因为我们没有调用定义在 module.py 中的 `_load_from_state_dict` 函数
                if torch.jit.is_scripting():
                    # 如果当前参数是 `scale`
                    if name == 'scale':
                        # 直接复制值给 `scale` 属性
                        self.scale.copy_(val)
                    else:
                        # 否则当前参数是 `zero_point`
                        assert name == 'zero_point'
                        # 直接复制值给 `zero_point` 属性
                        self.zero_point.copy_(val)
            # 如果 `strict` 为 True，并且未找到当前参数键
            elif strict:
                # 将当前键添加到 `missing_keys` 列表中
                missing_keys.append(key)
        # 调用父类的 `_load_from_state_dict` 方法，加载其余的状态字典项
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
class FixedQParamsFakeQuantize(FakeQuantize):
    """Simulate quantize and dequantize in training time.

    Simulate quantize and dequantize with fixed quantization
    parameters in training time. Only per tensor quantization
    is supported.
    """

    # TODO: rename observer to observer_ctr
    def __init__(self, observer):
        super().__init__(observer=observer)
        # 断言确保传入的观察器是 FixedQParamsObserver 类型
        assert type(self.activation_post_process) == FixedQParamsObserver, \
            f"{self.__class__.__name__}'s observer must be a {FixedQParamsObserver.__name__}"
        # 将 observer 参数赋给 _observer_ctr 属性
        self._observer_ctr = observer
        # 获取量化参数的 scale 和 zero_point
        self.scale = self.activation_post_process.scale
        self.zero_point = self.activation_post_process.zero_point
        # 断言检查是否为每个张量的量化方案
        assert _is_per_tensor(self.qscheme), 'Only per tensor quantization is supported' + \
            ' FixedQParamsFakeQuantize module, got qscheme:' + str(self.qscheme)

    @torch.jit.export
    def calculate_qparams(self):
        # 返回当前的量化参数 scale 和 zero_point
        return self.scale, self.zero_point

    @torch.jit.export
    def extra_repr(self):
        """Define a string representation of the object's attributes."""
        # 返回对象属性的字符串表示，包括量化相关的属性
        return f'fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, ' \
               f'scale={self.scale}, zero_point={self.zero_point}, ' \
               f'dtype={self.dtype}, quant_min={self.activation_post_process.quant_min}, ' \
               f'quant_max={self.activation_post_process.quant_max}, qscheme={self.qscheme}'


class FusedMovingAvgObsFakeQuantize(FakeQuantize):
    r"""Define a fused module to observe the tensor.

    Fused module that is used to observe the input tensor (compute min/max), compute
    scale/zero_point and fake_quantize the tensor.
    This module uses calculation similar MovingAverageMinMaxObserver for the inputs,
    to compute the min/max values in order to compute the scale/zero_point.
    The qscheme input in the observer is used to differentiate between symmetric/affine
    quantization scheme.

    The output of this module is given by
    x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale

    Similar to :class:`~torch.ao.quantization.FakeQuantize`, and accepts the same attributes as the
    base class.

    """

    def __init__(
        self,
        observer: Any = MovingAverageMinMaxObserver,
        quant_min: int = 0,
        quant_max: int = 255,
        **observer_kwargs: Any
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(observer, quant_min, quant_max, **observer_kwargs)
        # 断言确保 activation_post_process 是 MovingAverageMinMaxObserver 或 MovingAveragePerChannelMinMaxObserver 类型之一
        assert isinstance(self.activation_post_process, (MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver)), \
            "Fused observer+fake_quant module only works with MovingAverageMinMaxObserver"
        # 注册缓冲区，启用 fake_quant 和 observer
        self.register_buffer("fake_quant_enabled", torch.tensor([1], dtype=torch.long))
        self.register_buffer("observer_enabled", torch.tensor([1], dtype=torch.long))
        # 检查是否是对称量化方案
        self.is_symmetric_quant = _is_symmetric_quant(self.activation_post_process.qscheme)
    # 使用装饰器将该方法导出为 Torch JIT 模型的一部分，用于量化参数计算
    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # 调用激活后处理对象的 calculate_qparams 方法，返回量化参数的元组
        return self.activation_post_process.calculate_qparams()

    # 使用装饰器将该方法导出为 Torch JIT 模型的一部分，返回对象的字符串表示形式
    @torch.jit.export
    def extra_repr(self) -> str:
        # 返回对象的详细信息字符串，包括各种状态和参数
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"scale={self.scale}, zero_point={self.zero_point}, dtype={self.dtype}, "
            f"quant_min={self.activation_post_process.quant_min}, quant_max={self.activation_post_process.quant_max}, "
            f"qscheme={self.qscheme}, reduce_range={self.activation_post_process.reduce_range}"
        )

    # 此方法定义了 Torch 模型的前向传播过程
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # 调用 Torch 提供的融合运算函数，实现移动平均观察和伪量化
        return torch.fused_moving_avg_obs_fake_quant(
            X,
            self.observer_enabled,  # 是否启用观察器
            self.fake_quant_enabled,  # 是否启用伪量化
            self.activation_post_process.min_val,  # 激活后处理对象的最小值
            self.activation_post_process.max_val,  # 激活后处理对象的最大值
            self.scale,  # 量化的比例因子
            self.zero_point,  # 量化的零点
            self.activation_post_process.averaging_constant,  # 平均常数
            self.activation_post_process.quant_min,  # 量化的最小值
            self.activation_post_process.quant_max,  # 量化的最大值
            self.ch_axis,  # 通道轴
            self.is_per_channel,  # 是否按通道量化
            self.is_symmetric_quant,  # 是否对称量化
        )
# 定义默认的激活量化器 FakeQuantize，使用 MovingAverageMinMaxObserver 进行观察
# 设置量化的最小值和最大值为 0 到 255
# 数据类型为 torch.quint8，量化方案为 torch.per_tensor_affine，减少量化范围为 True
default_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
                                            dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=True)

# 定义默认的权重量化器 FakeQuantize，使用 MovingAverageMinMaxObserver 进行观察
# 设置量化的最小值和最大值为 -128 到 127
# 数据类型为 torch.qint8，量化方案为 torch.per_tensor_symmetric，不减少量化范围
default_weight_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127,
                                                   dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)

# 定义默认的动态激活量化器 FakeQuantize，使用 MovingAverageMinMaxObserver 进行观察
# 设置量化的最小值和最大值为 0 到 255，启用动态量化
# 数据类型为 torch.quint8，量化方案为 torch.per_tensor_affine，平均常数为 1
default_dynamic_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, is_dynamic=True,
    dtype=torch.quint8, averaging_constant=1)

# 使用 default_fixed_qparams_range_neg1to1_observer 定义范围为 -1 到 1 的固定参数量化器 FixedQParamsFakeQuantize
default_fixed_qparams_range_neg1to1_fake_quant = (
    FixedQParamsFakeQuantize.with_args(observer=default_fixed_qparams_range_neg1to1_observer)
)

# 使用 default_fixed_qparams_range_0to1_observer 定义范围为 0 到 1 的固定参数量化器 FixedQParamsFakeQuantize
default_fixed_qparams_range_0to1_fake_quant = (
    FixedQParamsFakeQuantize.with_args(observer=default_fixed_qparams_range_0to1_observer)
)

# 以下两个变量保留用于向后兼容性; 几个版本后移除
# 默认的对称固定参数量化器设置为范围为 -1 到 1 的固定参数量化器
default_symmetric_fixed_qparams_fake_quant = default_fixed_qparams_range_neg1to1_fake_quant

# 默认的仿射固定参数量化器设置为范围为 0 到 1 的固定参数量化器
default_affine_fixed_qparams_fake_quant = default_fixed_qparams_range_0to1_fake_quant

# 定义默认的每通道权重量化器 FakeQuantize，使用 MovingAveragePerChannelMinMaxObserver 进行观察
# 设置量化的最小值和最大值为 -128 到 127
# 数据类型为 torch.qint8，量化方案为 torch.per_channel_symmetric，不减少量化范围，通道轴为 0
default_per_channel_weight_fake_quant = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                               quant_min=-128,
                                                               quant_max=127,
                                                               dtype=torch.qint8,
                                                               qscheme=torch.per_channel_symmetric,
                                                               reduce_range=False,
                                                               ch_axis=0)

# 定义默认的嵌入量化器 FakeQuantize，使用 MovingAveragePerChannelMinMaxObserver 进行观察
# 设置量化方案为 torch.per_channel_affine_float_qparams
# 数据类型为 torch.quint8，量化的最小值和最大值为 0 到 255，通道轴为 0，平均常数为 1
default_embedding_fake_quant = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                      qscheme=torch.per_channel_affine_float_qparams,
                                                      dtype=torch.quint8,
                                                      quant_min=0,
                                                      quant_max=255,
                                                      ch_axis=0,
                                                      averaging_constant=1)
# 使用 FakeQuantize 创建一个用于 4 位假量化的默认设置，观察器为 MovingAveragePerChannelMinMaxObserver
# 使用 torch.per_channel_affine_float_qparams 方案对每个通道进行仿射浮点参数量化
# 通道轴设为 0，数据类型为 torch.quint4x2，平均常数为 1
default_embedding_fake_quant_4bit = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                           qscheme=torch.per_channel_affine_float_qparams,
                                                           ch_axis=0,
                                                           dtype=torch.quint4x2,
                                                           averaging_constant=1)

# 使用 FakeQuantize 创建一个用于直方图观察器的默认设置，量化范围为 0 到 255
# 数据类型为 torch.quint8，方案为 torch.per_tensor_affine，启用范围缩减
default_histogram_fake_quant = FakeQuantize.with_args(observer=HistogramObserver,
                                                      quant_min=0,
                                                      quant_max=255,
                                                      dtype=torch.quint8,
                                                      qscheme=torch.per_tensor_affine,
                                                      reduce_range=True)

# 使用 FusedMovingAvgObsFakeQuantize 创建融合版本的激活假量化的默认设置
# 观察器为 MovingAverageMinMaxObserver，量化范围为 0 到 255，数据类型为 torch.quint8
default_fused_act_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                       quant_min=0,
                                                                       quant_max=255,
                                                                       dtype=torch.quint8)

# 使用 FusedMovingAvgObsFakeQuantize 创建融合版本的权重假量化的默认设置
# 观察器为 MovingAverageMinMaxObserver，量化范围为 -128 到 127，数据类型为 torch.qint8
# 使用 torch.per_tensor_symmetric 方案对张量进行对称量化
default_fused_wt_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                      quant_min=-128,
                                                                      quant_max=127,
                                                                      dtype=torch.qint8,
                                                                      qscheme=torch.per_tensor_symmetric)

# 使用 FusedMovingAvgObsFakeQuantize 创建融合版本的通道权重假量化的默认设置
# 观察器为 MovingAveragePerChannelMinMaxObserver，量化范围为 -128 到 127，数据类型为 torch.qint8
# 使用 torch.per_channel_symmetric 方案对每个通道进行对称量化
default_fused_per_channel_wt_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                                                  quant_min=-128,
                                                                                  quant_max=127,
                                                                                  dtype=torch.qint8,
                                                                                  qscheme=torch.per_channel_symmetric)
# 创建一个使用移动平均最小-最大观察器的融合版本的假量化对象，限定8位值在[-127, +127]范围内（不包括-128）
fused_wt_fake_quant_range_neg_127_to_127 = FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                   quant_min=-127,
                                                                                   quant_max=127,
                                                                                   dtype=torch.qint8,
                                                                                   qscheme=torch.per_tensor_symmetric,
                                                                                   eps=2 ** -12)
"""
Fused version of `default_weight_fake_quant`, with the 8-bit values restricted to [-127, +127], excluding -128.
"""

# 创建一个使用移动平均每通道最小-最大观察器的融合版本的假量化对象，限定8位值在[-127, +127]范围内（不包括-128）
fused_per_channel_wt_fake_quant_range_neg_127_to_127 = \
    FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                            quant_min=-127,
                                            quant_max=127,
                                            dtype=torch.qint8,
                                            qscheme=torch.per_channel_symmetric,
                                            eps=2 ** -12)

"""
Fused version of `default_per_channel_weight_fake_quant`, with the 8-bit values restricted to [-127, +127], excluding -128.
"""

def _is_fake_quant_script_module(mod):
    """Return true if given mod is an instance of FakeQuantize script module."""
    # 如果给定的模块是 torch.jit.RecursiveScriptModule 的实例
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        # 获取限定名，例如 '__torch__.torch.ao.quantization.fake_quantize.___torch_mangle_2.FakeQuantize'
        suffix = mod._c.qualified_name.split('.', 1)[1]
        # 去掉后缀中的 '___torch_mangle_\d+'
        name = re.sub(r'\.___torch_mangle_\d+', '', suffix)
        # 判断是否是 FakeQuantize 或 FusedMovingAvgObsFakeQuantize
        return name == 'torch.ao.quantization.fake_quantize.FakeQuantize' or \
            name == 'torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize'
    return False

def disable_fake_quant(mod):
    """Disable fake quantization for the module.

    Disable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_fake_quant)

    """
    # 如果模块是 FakeQuantizeBase 的实例或者是 FakeQuantize 脚本模块
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        # 调用模块的 disable_fake_quant 方法
        mod.disable_fake_quant()

def enable_fake_quant(mod):
    """Enable fake quantization for the module.

    Enable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.enable_fake_quant)

    """
    # 如果模块是 FakeQuantizeBase 的实例或者是 FakeQuantize 脚本模块
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        # 调用模块的 enable_fake_quant 方法
        mod.enable_fake_quant()

def disable_observer(mod):
    """Disable observation for this module.

    Disable observation for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_observer)

    """

# 如果给定的模块是 FakeQuantizeBase 的实例或者是 FakeQuantize 脚本模块，则禁用该模块的假量化
def disable_observer(mod):
    """Disable observation for this module.

    Disable observation for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_observer)

    """
    # 如果 mod 是 FakeQuantizeBase 类的实例或者 _is_fake_quant_script_module(mod) 返回 True，
    # 则调用 mod 的 disable_observer() 方法来禁用观察器。
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.disable_observer()
def enable_observer(mod):
    """Enable observation for this module.

    Enable observation for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.enable_observer)

    """
    # 检查模块是否是 FakeQuantizeBase 类型或者其衍生类，或者是否是 FakeQuantize 脚本模块
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        # 如果是以上类型的模块，则调用其 enable_observer 方法
        mod.enable_observer()
```