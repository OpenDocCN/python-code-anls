# `.\pytorch\torch\ao\nn\qat\dynamic\modules\linear.py`

```
# 引入torch模块，mypy允许未类型化的定义
import torch

# 定义模块的公开接口列表
__all__ = ["Linear"]

# 定义一个继承自torch.ao.nn.qat.Linear的线性模块
class Linear(torch.ao.nn.qat.Linear):
    """
    一个线性模块，附带有用于权重的FakeQuantize模块，
    用于动态量化感知训练。

    我们采用与 `torch.nn.Linear` 相同的接口，请参见
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    获取文档。

    类似于 `torch.nn.Linear`，但初始化为默认的FakeQuantize模块。
    """

    def __init__(self, in_features, out_features, bias=True,
                 qconfig=None, device=None, dtype=None) -> None:
        # 调用父类的构造函数初始化线性模块
        super().__init__(in_features, out_features, bias, qconfig, device, dtype)
        
        # 如果不是内存无关的观察器，则引发值错误
        if not torch.ao.quantization.qconfig._activation_is_memoryless(qconfig):
            raise ValueError(
                "Dynamic QAT requires a memoryless observer." +
                "This means a MovingAverage observer with averaging constant equal to 1"
            )
```