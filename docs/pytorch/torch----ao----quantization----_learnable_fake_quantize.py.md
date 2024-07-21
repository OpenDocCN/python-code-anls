# `.\pytorch\torch\ao\quantization\_learnable_fake_quantize.py`

```py
# mypy: allow-untyped-defs
# 导入 torch 库
import torch
# 导入 Parameter 类
from torch.nn.parameter import Parameter
# 导入 List 类型
from typing import List

# 初始化全局变量 __all__ 为空列表
__all__: List[str] = []

# 定义 _LearnableFakeQuantize 类，继承自 torch.ao.quantization.FakeQuantizeBase 类
class _LearnableFakeQuantize(torch.ao.quantization.FakeQuantizeBase):
    r"""Generalized extension of the FakeQuantize module in fake_quantize.py.

    This is an extension of the FakeQuantize module in fake_quantize.py, which
    supports more generalized lower-bit quantization and supports learning of the scale
    and zero point parameters through backpropagation.

    In addition to the attributes in the original FakeQuantize module, the _LearnableFakeQuantize
    module also includes the following attributes to support quantization parameter learning.

    * :attr:`channel_len` defines the length of the channel when initializing scale and zero point
      for the per channel case.

    * :attr:`use_grad_scaling` defines the flag for whether the gradients for scale and zero point are
      normalized by the constant, which is proportional to the square root of the number of
      elements in the tensor. The related literature justifying the use of this particular constant
      can be found here: https://openreview.net/pdf?id=rkgO66VKDS.

    * :attr:`fake_quant_enabled` defines the flag for enabling fake quantization on the output.

    * :attr:`static_enabled` defines the flag for using observer's static estimation for
      scale and zero point.

    * :attr:`learning_enabled` defines the flag for enabling backpropagation for scale and zero point.
    """
    def __init__(self, observer, quant_min=0, quant_max=255, scale=1., zero_point=0., channel_len=-1,
                 use_grad_scaling=False, **observer_kwargs):
        super().__init__()
        # 断言确保量化的最小值小于最大值
        assert quant_min < quant_max, 'quant_min must be strictly less than quant_max.'
        self.quant_min = quant_min
        self.quant_max = quant_max
        # 将 quant_min 和 quant_max 参数传递给观察器的关键字参数
        observer_kwargs["quant_min"] = quant_min
        observer_kwargs["quant_max"] = quant_max
        self.use_grad_scaling = use_grad_scaling
        if channel_len == -1:
            # 如果 channel_len 为 -1，创建单一张量的参数 scale 和 zero_point
            self.scale = Parameter(torch.tensor([scale]))
            self.zero_point = Parameter(torch.tensor([zero_point]))
        else:
            # 断言确保 channel_len 是正整数
            assert isinstance(channel_len, int) and channel_len > 0, "Channel size must be a positive integer."
            # 创建长度为 channel_len 的 scale 和 zero_point 参数张量
            self.scale = Parameter(torch.tensor([scale] * channel_len))
            self.zero_point = Parameter(torch.tensor([zero_point] * channel_len))

        # 创建 activation_post_process 对象，使用给定的观察器和参数
        self.activation_post_process = observer(**observer_kwargs)
        # 断言确保 quant_min 在数据类型的范围内
        assert torch.iinfo(self.activation_post_process.dtype).min <= quant_min, \
            'quant_min out of bound'
        # 断言确保 quant_max 在数据类型的范围内
        assert quant_max <= torch.iinfo(self.activation_post_process.dtype).max, \
            'quant_max out of bound'
        # 记录数据类型、量化方案和通道轴信息（如果有的话）
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1
        # 注册三个标志位缓冲区，用于启用或禁用假量化、静态估计和学习
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('static_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('learning_enabled', torch.tensor([0], dtype=torch.uint8))

        # 计算量化位宽
        bitrange = torch.tensor(quant_max - quant_min + 1).double()
        self.bitwidth = int(torch.log2(bitrange).item())
        # 注册 eps 缓冲区，用于存储浮点数精度的极小值
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))

    @torch.jit.export
    def enable_param_learning(self):
        r"""Enable parameter learning over static observer estimates.

        Enables learning of quantization parameters and
        disables static observer estimates. Forward path returns fake quantized X.
        """
        # 启用参数学习，禁用静态估计，并启用假量化
        self.toggle_qparam_learning(enabled=True) \
            .toggle_fake_quant(enabled=True) \
            .toggle_observer_update(enabled=False)
        return self

    @torch.jit.export
    def enable_static_estimate(self):
        """Enable static estimates of quantization parameters.

        Enables static observer estimates and disables learning of
        quantization parameters. Forward path returns fake quantized X.
        """
        # 启用静态估计，禁用参数学习，并启用假量化
        self.toggle_qparam_learning(enabled=False) \
            .toggle_fake_quant(enabled=True) \
            .toggle_observer_update(enabled=True)

    @torch.jit.export
    # 启用静态观察，累积数据但不更新量化参数
    # 此方法使静态观察器从输入累积数据，但不会更新量化参数。前向路径返回原始输入 X。
    def enable_static_observation(self):
        # 禁用量化参数学习和假量化
        self.toggle_qparam_learning(enabled=False) \
            .toggle_fake_quant(enabled=False) \
            .toggle_observer_update(enabled=True)

    @torch.jit.export
    def toggle_observer_update(self, enabled=True):
        # 更新静态观察器的状态
        self.static_enabled[0] = int(enabled)  # type: ignore[operator]
        return self

    @torch.jit.export
    def enable_observer(self, enabled=True):
        # 启用或禁用观察器
        self.toggle_observer_update(enabled)

    @torch.jit.export
    def toggle_qparam_learning(self, enabled=True):
        # 切换量化参数学习状态
        self.learning_enabled[0] = int(enabled)  # type: ignore[operator]
        self.scale.requires_grad = enabled
        self.zero_point.requires_grad = enabled
        return self

    @torch.jit.export
    def toggle_fake_quant(self, enabled=True):
        # 切换假量化状态
        self.fake_quant_enabled[0] = int(enabled)
        return self

    @torch.jit.export
    def observe_quant_params(self):
        # 打印当前量化参数的信息
        print(f'_LearnableFakeQuantize Scale: {self.scale.detach()}')
        print(f'_LearnableFakeQuantize Zero Point: {self.zero_point.detach()}')

    @torch.jit.export
    def calculate_qparams(self):
        # 计算并返回量化参数
        self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]
        scale = self.scale.detach()
        zero_point = self.zero_point.detach().round().clamp(self.quant_min, self.quant_max).long()
        return scale, zero_point
    # 定义前向传播函数，接收输入张量 X
    def forward(self, X):
        # 如果静态量化被启用
        if self.static_enabled[0] == 1:  # type: ignore[index]
            # 对输入 X 进行离线量化后处理
            self.activation_post_process(X.detach())
            # 计算量化参数的缩放因子和零点
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            # 将缩放因子和零点移到与当前设备相同的设备上
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)
            # 将计算得到的缩放因子和零点复制给当前对象的缩放因子和零点
            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point)
        else:
            # 如果静态量化未启用，则对缩放因子进行最小值约束
            self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]

        # 如果伪量化被启用
        if self.fake_quant_enabled[0] == 1:
            # 如果量化方案是每通道对称或每张量对称，则将零点设为零
            if self.qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric):
                self.zero_point.data.zero_()

            # 如果使用梯度缩放
            if self.use_grad_scaling:
                grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
            else:
                grad_factor = 1.0
            
            # 根据量化方案选择相应的伪量化函数进行处理
            if self.qscheme in (
                    torch.per_channel_symmetric, torch.per_channel_affine):
                X = torch._fake_quantize_learnable_per_channel_affine(
                    X, self.scale, self.zero_point, self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                X = torch._fake_quantize_learnable_per_tensor_affine(
                    X, self.scale, self.zero_point,
                    self.quant_min, self.quant_max, grad_factor)

        # 返回处理后的张量 X
        return X
```