# `.\pytorch\torch\ao\nn\quantized\modules\batchnorm.py`

```
# mypy: allow-untyped-defs
# 引入PyTorch和PyTorch的AO（自动优化）模块中的intrinsic模块
import torch
import torch.ao.nn.intrinsic as nni

# 定义模块的公开接口列表
__all__ = [
    "BatchNorm2d",
    "BatchNorm3d"
]

# 定义一个继承自torch.nn.modules.batchnorm._BatchNorm的内部类
class _BatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类的初始化方法，设置批标准化的参数和缓冲区
        super().__init__(num_features, eps, momentum, True, True, **factory_kwargs)
        # 注册scale和zero_point两个缓冲区
        self.register_buffer('scale', torch.tensor(1.0, **factory_kwargs))
        self.register_buffer('zero_point', torch.tensor(0, **factory_kwargs))

    @staticmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 获取激活后处理的实例
        activation_post_process = mod.activation_post_process
        # 如果mod是_NNI_BN_RELU_MODULE类型的，转换为mod[0]
        if type(mod) == cls._NNI_BN_RELU_MODULE:
            mod = mod[0]
        # 计算量化参数的scale和zero_point
        scale, zero_point = activation_post_process.calculate_qparams()
        # 创建一个新的量化模块实例
        new_mod = cls(mod.num_features, mod.eps)
        # 复制权重、偏置、均值、方差以及量化参数scale和zero_point
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.running_mean = mod.running_mean
        new_mod.running_var = mod.running_var
        new_mod.scale = scale
        new_mod.zero_point = zero_point
        return new_mod

    @classmethod
    def from_reference(cls, bn, output_scale, output_zero_point):
        # 使用参考批标准化bn的参数创建一个新的量化批标准化实例qbn
        qbn = cls(
            bn.num_features,
            bn.eps,
            bn.momentum,
            device=bn.weight.device,
            dtype=bn.weight.dtype
        )
        qbn.weight = bn.weight
        qbn.bias = bn.bias
        qbn.running_mean = bn.running_mean
        qbn.running_var = bn.running_var
        qbn.scale = output_scale
        qbn.zero_point = output_zero_point
        return qbn

# 继承自_BatchNorm的BatchNorm2d类，用于实现二维量化批标准化
class BatchNorm2d(_BatchNorm):
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm2d`.
    """

    # 定义_NNI_BN_RELU_MODULE为nni.BNReLU2d，用于处理带ReLU的批标准化
    _NNI_BN_RELU_MODULE = nni.BNReLU2d

    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类_BatchNorm的初始化方法
        super().__init__(num_features, eps, momentum, **factory_kwargs)

    def _get_name(self):
        # 返回该类的名称
        return 'QuantizedBatchNorm2d'

    def _check_input_dim(self, input):
        # 临时使用len(shape)来检查输入维度是否为4，而不是ndim，因为存在JIT问题
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 禁用此处代码的追踪，因为它不是符号追踪的一部分
        # self._check_input_dim(input)
        # 调用torch.ops.quantized.batch_norm2d进行二维量化批标准化的前向传播
        return torch.ops.quantized.batch_norm2d(
            input, self.weight, self.bias, self.running_mean,
            self.running_var, self.eps, self.scale, self.zero_point)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 调用_BatchNorm类的from_float方法，返回量化模块的实例
        return _BatchNorm.from_float(cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant)

# 继承自_BatchNorm的BatchNorm3d类，用于实现三维量化批标准化
class BatchNorm3d(_BatchNorm):
    # 留待实现
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm3d`.
    """

    # 定义一个全局变量，指向 nninit 库中的 BNReLU3d 类
    _NNI_BN_RELU_MODULE = nni.BNReLU3d

    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None):
        # 准备传递给父类构造函数的关键字参数字典
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类的构造函数初始化 BatchNorm3d 对象
        super().__init__(num_features, eps, momentum, **factory_kwargs)

    def _get_name(self):
        # 返回当前类的名称作为字符串
        return 'QuantizedBatchNorm3d'

    def _check_input_dim(self, input):
        # 检查输入张量的维度是否为 5，暂时使用 len(shape) 而非 ndim，因为 JIT 存在问题
        # 参考：https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, H, W)`!")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 由于此部分无法在符号跟踪中使用，因此禁用该检查
        # self._check_input_dim(input)
        # 调用量化版本的 batch_norm3d 操作进行前向传播
        return torch.ops.quantized.batch_norm3d(
            input, self.weight, self.bias, self.running_mean,
            self.running_var, self.eps, self.scale, self.zero_point)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 调用内部方法 _BatchNorm.from_float 以创建一个量化的 BatchNorm3d 对象
        return _BatchNorm.from_float(cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant)
```