# `.\pytorch\torch\ao\quantization\experimental\fake_quantize.py`

```
# mypy: allow-untyped-defs
# 导入 torch 库
import torch
# 从 torch 库导入 Tensor 类型
from torch import Tensor
# 从 torch.ao.quantization.experimental.observer 模块导入 APoTObserver 类
from torch.ao.quantization.experimental.observer import APoTObserver
# 从 torch.ao.quantization.fake_quantize 模块导入 FakeQuantizeBase 类
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
# 从 torch.ao.quantization.experimental.fake_quantize_function 模块导入 fake_quantize_function 函数
from torch.ao.quantization.experimental.fake_quantize_function import fake_quantize_function

# 定义 APoTFakeQuantize 类，继承自 FakeQuantizeBase 类
class APoTFakeQuantize(FakeQuantizeBase):
    # 类型注解，alpha 属性为 Tensor 类型
    alpha: Tensor
    # 类型注解，gamma 属性为 Tensor 类型
    gamma: Tensor
    # 类型注解，quantization_levels 属性为 Tensor 类型
    quantization_levels: Tensor
    # 类型注解，level_indices 属性为 Tensor 类型
    level_indices: Tensor

    # 构造方法，接受 observer 类型和其他关键字参数
    def __init__(self, observer=APoTObserver, **observer_kwargs):
        super().__init__()  # 调用父类的构造方法
        # 创建观察器实例并存储在属性 activation_post_process 中
        self.activation_post_process = observer(**observer_kwargs)
        # 设置属性 dtype 为 activation_post_process 的数据类型

    # 定义 calculate_qparams 方法，计算量化参数，支持 signed 参数
    def calculate_qparams(self, signed=False):  # type: ignore[override]
        return self.activation_post_process.calculate_qparams(signed=signed)

    # 重写父类的 forward 方法，处理输入张量 X
    def forward(self, X: torch.Tensor):  # type: ignore[override]
        # 如果观察器已启用
        if self.observer_enabled[0] == 1:
            # 对输入 X 进行前向传播处理
            self.activation_post_process.forward(X)
            # 计算量化参数，并分别赋值给 alpha、gamma、quantization_levels、level_indices 属性
            result = self.activation_post_process.calculate_qparams(signed=False)
            self.alpha = result[0]
            self.gamma = result[1]
            self.quantization_levels = result[2]
            self.level_indices = result[3]

        # 如果假量化已启用
        if self.fake_quant_enabled[0] == 1:
            # 断言确保 alpha、gamma、quantization_levels、level_indices 四个属性都不为 None
            assert (self.alpha is not None
                    and self.gamma is not None
                    and self.quantization_levels is not None
                    and self.level_indices is not None), "Must set qparams for fake quant"

            # 应用假量化函数 fake_quantize_function 到输入 X 上
            X = fake_quantize_function.apply(X, self.alpha, self.gamma, self.quantization_levels, self.level_indices)

        # 返回处理后的张量 X
        return X
```