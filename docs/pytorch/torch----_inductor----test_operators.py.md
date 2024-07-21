# `.\pytorch\torch\_inductor\test_operators.py`

```
# mypy: allow-untyped-defs
# 导入 torch 库的 `library` 模块
import torch.library
# 导入 Tensor 类型
from torch import Tensor
# 导入 Function 类
from torch.autograd import Function

# 如果不是在部署模式下运行
if not torch._running_with_deploy():
    # 创建一个名为 `_test_lib_def` 的 Torch 库对象，用于定义接口
    _test_lib_def = torch.library.Library("_inductor_test", "DEF")
    # 定义一个名为 `realize` 的函数签名，接受一个 Tensor 参数，并使用 pt2_compliant_tag 标签
    _test_lib_def.define(
        "realize(Tensor self) -> Tensor", tags=torch.Tag.pt2_compliant_tag
    )

    # 创建一个名为 `_test_lib_impl` 的 Torch 库对象，用于实现接口
    _test_lib_impl = torch.library.Library("_inductor_test", "IMPL")
    # 对于每个调度键（dispatch_key）如 "CPU", "CUDA", "Meta"
    for dispatch_key in ("CPU", "CUDA", "Meta"):
        # 实现 `realize` 函数，针对不同的调度键，使用 lambda 函数返回输入的克隆（clone）
        _test_lib_impl.impl("realize", lambda x: x.clone(), dispatch_key)

    # 定义一个名为 `Realize` 的 Torch Function 类
    class Realize(Function):
        # 静态方法：前向传播函数，调用 `_inductor_test.realize` 运算符
        @staticmethod
        def forward(ctx, x):
            return torch.ops._inductor_test.realize(x)

        # 静态方法：反向传播函数，直接返回梯度输出
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    # 定义一个名为 `realize` 的函数，接受一个 Tensor 参数，并应用 `Realize` 的实现
    def realize(x: Tensor) -> Tensor:
        return Realize.apply(x)
```