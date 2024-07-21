# `.\pytorch\torch\ao\quantization\experimental\adaround_fake_quantize.py`

```
# mypy: allow-untyped-defs
from typing import Tuple

import torch  # 导入 PyTorch 库
from torch.ao.quantization.fake_quantize import _is_symmetric_quant  # 导入用于检查是否对称量化的函数
from torch.ao.quantization.utils import is_per_tensor  # 导入用于检查是否为每个张量量化的函数
from torch.quantization import FakeQuantize  # 导入 FakeQuantize 类
from torch.quantization.observer import MinMaxObserver  # 导入 MinMaxObserver 类


class AdaroundFakeQuantizer(FakeQuantize):
    """
    This is a FakeQuantizer that enables an adaptive rounding fake quantizer.
    Adaround is a technique to adaptively round weights, derived from the paper https://arxiv.org/pdf/2004.10568.pdf
    For HTP compatibility, we are targeting to use symmetric quantization
    """

    scale: torch.Tensor  # 缩放因子
    zero_point: torch.Tensor  # 零点
    V: torch.nn.Parameter  # 参数 V

    # pyre-fixme[3]: Return type must be annotated.
    def __init__(
        self,
        observer=MinMaxObserver,  # 观察器，默认为 MinMaxObserver
        qscheme=torch.per_tensor_symmetric,  # 量化方案，默认为对称量化，不被使用，但需要用于 fakequant
        quant_min: int = -128,  # 最小量化值，默认为 -128
        quant_max: int = 127,  # 最大量化值，默认为 127
        ch_axis: int = 0,  # 通道轴，默认为 0
        # pyre-fixme[2]: Parameter must be annotated.
        **observer_kwargs,  # 其他观察器参数
    ):
        super().__init__(
            observer=observer,  # 调用父类构造函数初始化观察器
            qscheme=qscheme,  # 传递量化方案给父类构造函数
            quant_min=quant_min,  # 传递最小量化值给父类构造函数
            quant_max=quant_max,  # 传递最大量化值给父类构造函数
            is_dynamic=False,  # 不启用动态量化
            **observer_kwargs,  # 传递其他观察器参数给父类构造函数
        )
        # 如果 quant_min 和 quant_max 合法，则将其加入 observer_kwargs
        if quant_min is not None and quant_max is not None:
            assert (
                quant_min <= quant_max
            ), "quant_min must be less than or equal to quant_max"
        # pyre-fixme[4]: Attribute must be annotated.
        self.qscheme = qscheme  # 设置量化方案
        self.is_per_tensor: bool = is_per_tensor(qscheme)  # 检查是否为每个张量量化
        self.is_symmetric: bool = _is_symmetric_quant(qscheme)  # 检查是否为对称量化
        assert self.is_symmetric, "Only symmetric quantization is supported"  # 断言只支持对称量化
        self.ch_axis: int = ch_axis  # 设置通道轴

        self.scale = torch.tensor([], requires_grad=False)  # 初始化缩放因子张量
        self.zero_point = torch.tensor([], requires_grad=False)  # 初始化零点张量
        self.V = torch.nn.Parameter(torch.tensor([]), requires_grad=True)  # 初始化参数 V 作为可训练参数
        # 固定 Stretch 参数
        self.zeta: torch.Tensor = torch.tensor(1.1, requires_grad=False)  # 设置 zeta 参数
        self.gamma: torch.Tensor = torch.tensor(-0.1, requires_grad=False)  # 设置 gamma 参数
        self.sigmoid = torch.nn.Sigmoid()  # 初始化 Sigmoid 激活函数
        self.use_soft_rounding = True  # 使用软量化

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.scale, self.zero_point  # 返回缩放因子和零点

    @torch.jit.export
    def extra_repr(self) -> str:
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"quant_min={self.activation_post_process.quant_min}, quant_max={self.activation_post_process.quant_max}, "
            f"dtype={self.dtype}, qscheme={self.qscheme}, ch_axis={self.ch_axis}, "
            f"scale={self.scale}, zero_point={self.zero_point}, (self.V >= 0).int().sum()={(self.V >= 0).int().sum()}"
        )  # 返回用于打印对象信息的字符串表示
    # 将 fake_quant_enabled 的第一个元素设置为 1，表示启用量化仿真
    def enable_weight_fake_quant(self) -> None:
        self.fake_quant_enabled[0] = 1

    # 获取修正的 sigmoid 函数
    def get_rectified_sigmoid_func(self) -> torch.Tensor:
        # 如果使用软舍入
        if self.use_soft_rounding:
            # 计算修正后的 sigmoid 函数值，并限制在 [0, 1] 范围内
            return torch.clamp(
                self.sigmoid(self.V) * (self.zeta - self.gamma) + self.gamma,
                min=0,
                max=1,
            )
        else:
            # 如果不使用软舍入，返回 V 是否大于等于 0 的整数值（0 或 1）
            return (self.V >= 0).int()

    # 忽略此函数的 Torch JIT 编译
    @torch.jit.ignore
    def update_scale(
        self, X: torch.Tensor, _scale: torch.Tensor, _zero_point: torch.Tensor
    ) -> None:
        # 如果 scale 的元素个数为 0，则将 _scale 和 _zero_point 数据分别赋值给 scale 和 zero_point
        if self.scale.numel() == 0:
            self.scale.data = _scale.to(X.device)
            self.zero_point = _zero_point.to(X.device)
        else:
            # 否则直接赋值 _scale 给 scale
            self.scale.data = _scale
            # 如果不是对称量化，则直接赋值 _zero_point 给 zero_point；否则将 zero_point 初始化为与 _zero_point 形状相同的全零张量
            if not self.is_symmetric:
                self.zero_point = _zero_point
            else:
                self.zero_point = torch.zeros_like(_zero_point)
            # 在除了通道轴之外的每个维度上，扩展 zero_point 的维度
            for i in range(X.dim()):
                if i == self.ch_axis:
                    continue
                self.zero_point = self.zero_point.unsqueeze(i)
        # 对输入张量 X 进行量化操作，计算量化后的值
        X_q = X / self.scale
        # 对量化后的值取下限，得到整数部分
        X_q_floor = torch.floor(X_q)
        # 计算残差，即量化后的值减去整数部分，其范围为 [0, 1)
        residual = X_q - X_q_floor
        # 断言残差应该非负且在 [0, 1) 范围内
        assert torch.all(
            torch.ge(residual, 0)
        ), "residual should be non-negative [0, 1)"
        # 初始化 V 的值，通过计算得到初始值，用于后续计算
        V_init = -torch.log((self.zeta - self.gamma) / (residual - self.gamma) - 1)
        # 将计算得到的初始值赋值给 V
        self.V.data = V_init
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # 如果开启了观察者模式
        if self.observer_enabled[0] == 1:
            # 分离输入张量 X 的副本
            X_detached = X.detach()
            # 对分离后的张量进行激活后处理
            self.activation_post_process(X_detached)
            # 计算激活后处理的量化参数
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            # 将量化参数转移到适当的设备上
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(
                self.zero_point.device
            )
            # 获取张量 X 的维度列表
            dims = list(range(X.dim()))
            # 如果不是每个张量都有独立的量化参数
            if not self.is_per_tensor:
                # 移除通道轴
                dims.remove(self.ch_axis)
            # 如果不是每个张量都有独立的量化参数
            if not self.is_per_tensor:
                # 遍历张量的维度
                for i in range(X.dim()):
                    # 如果当前维度是通道轴，则跳过
                    if i == self.ch_axis:
                        continue
                    # 在指定维度上增加 _scale 和 _zero_point
                    _scale = _scale.unsqueeze(i)
                    _zero_point = _zero_point.unsqueeze(i)
            # 更新量化的缩放和零点参数到观察者对象中
            self.update_scale(X_detached, _scale, _zero_point)

        # 如果开启了伪量化模式
        if self.fake_quant_enabled[0] == 1:
            # 执行软量化
            # 参考 Adaround 论文中的方程 (23)
            h_v = self.get_rectified_sigmoid_func()
            X_q = X / self.scale
            # 使用直通估计器对 floor 函数进行处理
            X_q_floor = torch.floor(X_q) + self.zero_point
            # 不考虑四舍五入，梯度应能够从 X_q_dq 回流到 self.V
            # 在 adaround 中，我们只训练 V，而不是权重
            X_q_dq = (
                torch.clamp(X_q_floor + h_v, min=self.quant_min, max=self.quant_max)
                - self.zero_point
            ) * self.scale
            return X_q_dq
        else:
            # 如果未开启伪量化模式，则直接返回输入张量 X
            return X
```